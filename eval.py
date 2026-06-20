import os
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import config
import utils
import data_loader

def main():
    # 1. Setup seed and device
    utils.set_seed(config.SEED)
    device = utils.get_device()
    print(f"Using device: {device}")
    
    # 2. Check model existence
    if not os.path.exists(config.FINAL_MODEL_DIR):
        print(f"Error: Trained model not found at {config.FINAL_MODEL_DIR}.")
        print("Please run train.py first to fine-tune the model.")
        return

    # 3. Load Model and Tokenizer
    print(f"Loading trained model from: {config.FINAL_MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(config.FINAL_MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.FINAL_MODEL_DIR)

    # 4. Load Dataset
    try:
        dataset_dict, test_df = data_loader.load_and_preprocess_data()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check your DATA_PATH in config.py.")
        return

    tokenized_datasets = data_loader.tokenize_datasets(dataset_dict, tokenizer)

    # 5. Create Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    # 6. Load Metrics
    rouge_metric, bleu_metric = utils.load_metrics()
    compute_metrics_fn = utils.get_compute_metrics_fn(tokenizer, rouge_metric, bleu_metric)

    # 7. Setup Trainer for prediction
    training_args = config.get_training_args(device)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn
    )

    # 8. Generate Predictions
    print("Generating predictions on the test set...")
    predict_results = trainer.predict(tokenized_datasets['test'])
    
    preds = predict_results.predictions
    labels = predict_results.label_ids

    # Handle shape conversions (e.g. tuples or logits)
    if isinstance(preds, tuple):
        preds = preds[0]
    if len(preds.shape) == 3:
        print("Converting logits to token IDs using argmax...")
        preds = np.argmax(preds, axis=-1)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    preds = np.where(preds != -100, preds, pad_token_id).astype(int)
    labels = np.where(labels != -100, labels, pad_token_id).astype(int)

    # 9. Decode Texts
    print("Decoding predictions...")
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # 10. Compute Metrics
    try:
        final_metrics = compute_metrics_fn((preds, labels))
        print("\nFinal Test Metrics:")
        for metric_name, score in final_metrics.items():
            print(f"  {metric_name}: {score:.4f}")
    except Exception as e:
        print(f"\nCould not compute final metrics directly: {e}")

    # 11. Create results DataFrame and compute similarity
    results_df = pd.DataFrame({
        'reference': decoded_labels,
        'generated': decoded_preds,
        'summary': test_df['summary'].values[:len(decoded_preds)]
    })

    results_df['similarity'] = results_df.apply(
        lambda row: utils.similarity_score(row['reference'], row['generated']), axis=1
    )

    # Categorize predictions based on similarity score
    results_df['category'] = pd.cut(
        results_df['similarity'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low Match', 'Medium Match', 'High Match']
    )

    print("\nQuality Distribution:")
    print(results_df['category'].value_counts().sort_index())

    # Save predictions
    output_predictions_path = 'test_predictions.csv'
    results_df.to_csv(output_predictions_path, index=False)
    print(f"\nResults saved to '{output_predictions_path}'")

    # Show some examples
    print("\nSample Predictions:")
    print("=" * 80)
    for i in range(min(3, len(results_df))):
        print(f"Example {i+1}:")
        print(f"  Summary:   {results_df.iloc[i]['summary'][:120]}...")
        print(f"  Reference: {results_df.iloc[i]['reference']}")
        print(f"  Generated: {results_df.iloc[i]['generated']}")
        print(f"  Similarity Score: {results_df.iloc[i]['similarity']:.2f} ({results_df.iloc[i]['category']})")
        print("-" * 80)

if __name__ == '__main__':
    main()
