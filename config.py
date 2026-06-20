import os
import torch
from transformers import Seq2SeqTrainingArguments

# Random Seed
SEED = 42

# Model configuration
MODEL_NAME = "t5-base"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 64

# Dataset configuration
DATA_PATH = './arXiv_scientific dataset.csv'  # Default path, can be updated as needed
SAMPLE_SIZE = 30000

# Text length filters (in words)
MIN_SUMMARY_LENGTH = 20
MAX_SUMMARY_LENGTH = 512
MIN_TITLE_LENGTH = 3
MAX_TITLE_LENGTH = 50

# Output directories
OUTPUT_DIR = "./results"
FINAL_MODEL_DIR = "./final_model"

# Training configuration
def get_training_args(device):
    return Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=device.type == "cuda",
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=100,
        report_to="none",
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=4,
        remove_unused_columns=True,
        push_to_hub=False,
        seed=SEED,
    )
