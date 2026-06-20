import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
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
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
    # 2. Load and Preprocess Data
    try:
        dataset_dict, _ = data_loader.load_and_preprocess_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please check your DATA_PATH in config.py.")
        return

    # 3. Load Model and Tokenizer
    print(f"Loading model: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_NAME)
    print(f"Model parameters: {model.num_parameters():,}")

    # 4. Tokenize Data
    tokenized_datasets = data_loader.tokenize_datasets(dataset_dict, tokenizer)

    # 5. Create Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    # 6. Load Metrics & Configure Compute Metrics
    rouge_metric, bleu_metric = utils.load_metrics()
    compute_metrics_fn = utils.get_compute_metrics_fn(tokenizer, rouge_metric, bleu_metric)

    # 7. Get Training Arguments
    training_args = config.get_training_args(device)
    print("Training Configuration Setup Completed.")

    # 8. Create Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn
    )

    # 9. Train the Model
    print("Starting training...")
    train_result = trainer.train()

    # 10. Save Model
    print(f"Saving final model to: {config.FINAL_MODEL_DIR}")
    trainer.save_model(config.FINAL_MODEL_DIR)
    tokenizer.save_pretrained(config.FINAL_MODEL_DIR)
    print(f"Training completed! Training loss: {train_result.training_loss:.4f}")

    # 11. Plot & Save Curves
    history_df = pd.DataFrame(trainer.state.log_history)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    history_df.to_csv(os.path.join(config.OUTPUT_DIR, "training_history.csv"), index=False)

    if 'loss' in history_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Training loss
        train_logs = history_df[history_df['loss'].notna()]
        if not train_logs.empty:
            axes[0].plot(train_logs['step'], train_logs['loss'], marker='o', label='Train Loss')
            axes[0].set_xlabel('Steps')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        # Validation loss
        eval_logs = history_df[history_df['eval_loss'].notna()]
        if not eval_logs.empty:
            axes[1].plot(eval_logs['step'], eval_logs['eval_loss'], marker='s', color='orange', label='Val Loss')
            axes[1].set_xlabel('Steps')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Validation Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(config.OUTPUT_DIR, "loss_curves.png")
        plt.savefig(plot_path)
        print(f"Loss curves saved to: {plot_path}")

if __name__ == '__main__':
    main()
