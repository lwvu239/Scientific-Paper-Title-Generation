import random
import torch
import numpy as np
import evaluate
from difflib import SequenceMatcher

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def similarity_score(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def load_metrics():
    # Load metrics from Hugging Face evaluate
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("sacrebleu")
    return rouge_metric, bleu_metric

def get_compute_metrics_fn(tokenizer, rouge_metric, bleu_metric):
    """
    Returns a compute_metrics function that closes over tokenizer and metrics modules.
    """
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # Ensure predictions are numpy arrays
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = np.array(predictions)
        labels = np.array(labels)

        # Replace -100 in labels with pad_token_id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Clip predictions to valid token range to avoid overflow
        vocab_size = tokenizer.vocab_size
        predictions = np.clip(predictions, 0, vocab_size - 1)
        labels = np.clip(labels, 0, vocab_size - 1)

        # Decode predictions and labels
        try:
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        except Exception as e:
            print(f"Decode error: {e}")
            return {
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'bleu': 0.0
            }

        # Clean up the decoded texts
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # Filter out empty predictions
        valid_pairs = [(p, l) for p, l in zip(decoded_preds, decoded_labels) if p and l]

        if not valid_pairs:
            return {
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'bleu': 0.0
            }

        decoded_preds, decoded_labels = zip(*valid_pairs)

        # Compute ROUGE
        try:
            rouge_result = rouge_metric.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                use_stemmer=True
            )
            rouge1 = rouge_result['rouge1']
            rouge2 = rouge_result['rouge2']
            rougeL = rouge_result['rougeL']
        except Exception as e:
            print(f"ROUGE computation error: {e}")
            rouge1 = rouge2 = rougeL = 0.0

        # Compute BLEU
        try:
            bleu_result = bleu_metric.compute(
                predictions=decoded_preds,
                references=[[label] for label in decoded_labels]
            )
            bleu_score = bleu_result['score']
        except Exception as e:
            print(f"BLEU computation error: {e}")
            bleu_score = 0.0

        result = {
            'rouge1': rouge1,
            'rouge2': rouge2,
            'rougeL': rougeL,
            'bleu': bleu_score
        }

        return result
    return compute_metrics
