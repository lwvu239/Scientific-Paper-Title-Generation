import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import config
import utils

class TitleGenerator:
    """
    TitleGenerator handles loading a fine-tuned or base Seq2Seq model (like T5)
    and generating scientific paper titles given their abstracts.
    """
    def __init__(self, model_path=config.FINAL_MODEL_DIR):
        self.device = utils.get_device()
        print(f"Loading inference model from '{model_path}' on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure the model is trained or point to a valid pre-trained model path.")
            raise e

    def generate(self, summary_text, num_beams=4, max_length=64, num_return_sequences=1):
        """
        Generate title(s) from a given summary (abstract).
        """
        # Prepend the task prefix for T5
        input_text = "summarize: " + summary_text
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=1.0
            )
            
        titles = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return titles

if __name__ == '__main__':
    # Default model path config
    model_path = config.FINAL_MODEL_DIR
    
    # If the fine-tuned model hasn't been trained/saved yet, fallback to the base model for a live test.
    if not os.path.exists(model_path):
        print(f"Fine-tuned model at '{model_path}' not found.")
        print(f"Falling back to base pre-trained model '{config.MODEL_NAME}' for testing...")
        model_path = config.MODEL_NAME
        
    try:
        generator = TitleGenerator(model_path)
    except Exception as e:
        print(f"Failed to initialize generator: {e}")
        sys.exit(1)
        
    # Sample abstract for test
    demo_abstract = """
    We present a novel approach to natural language processing using transformer-based
    architectures. Our method achieves state-of-the-art results on multiple benchmarks
    while being more computationally efficient than previous approaches.
    """
    
    print("\n" + "="*50)
    print("Input Abstract:")
    print(demo_abstract.strip())
    print("-" * 50)
    
    print("Generating Titles (3 candidate options):")
    try:
        titles = generator.generate(demo_abstract, num_beams=5, num_return_sequences=3)
        for i, title in enumerate(titles):
            print(f"  {i+1}. {title}")
    except Exception as e:
        print(f"Generation error: {e}")
    print("="*50 + "\n")
