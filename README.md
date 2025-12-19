# Scientific Paper Title Generation

Fine-tune a pre-trained encoder-decoder language model (T5) to generate compelling titles for scientific papers given their abstracts. This is an abstractive text summarization task using the arXiv Dataset.

##  Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Setup Instructions](#setup-instructions)
- [Reproduction Steps](#reproduction-steps)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Usage](#usage)

##  Overview

This project implements a scientific paper title generation system by fine-tuning the T5 (Text-to-Text Transfer Transformer) model. Given an abstract/summary of a scientific paper, the model generates an appropriate title.

**Key Features:**
- Fine-tuning T5-base model on 30,000 scientific paper samples
- ROUGE and BLEU evaluation metrics
- Interactive title generation function
- Qualitative analysis with similarity scoring

##  Requirements

### Python Version
- Python 3.8+

### Dependencies
```
transformers
datasets
evaluate
rouge-score
bert-score
sacrebleu
sentencepiece
accelerate
torch
pandas
numpy
matplotlib
seaborn
tqdm
```

### Hardware
- **Recommended:** GPU with CUDA support (NVIDIA GPU with 8GB+ VRAM)
- **Minimum:** CPU (training will be significantly slower)

##  Dataset

**Dataset:** [arXiv Scientific Dataset](https://drive.google.com/drive/folders/1b8JznRVbkYom1iIZFoBErdfksYsPk6kw)

The dataset contains metadata for scientific papers from arXiv, including:
- `summary`: Paper abstract (input)
- `title`: Paper title (target)

### Data Preprocessing
- **Sample Size:** 30,000 papers
- **Filtering Criteria:**
  - Summary length: 20-512 words
  - Title length: 3-50 words
- **Split Ratio:** 80% Train / 10% Validation / 10% Test

##  Setup Instructions

### 1. Clone/Download the Repository
```bash
git clone <repository-url>
cd Final
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install transformers datasets evaluate rouge-score bert-score sacrebleu sentencepiece accelerate torch pandas numpy matplotlib seaborn tqdm
```

### 4. Download the Dataset
1. Download the arXiv dataset from the [Google Drive link](https://drive.google.com/drive/folders/1b8JznRVbkYom1iIZFoBErdfksYsPk6kw)
2. Place `arXiv_scientific dataset.csv` in your Google Drive or local directory
3. Update the `data_path` variable in the notebook to point to your dataset location

## 🔄 Reproduction Steps

### Running on Google Colab (Recommended)

1. **Upload the notebook** to Google Colab
2. **Mount Google Drive** (the notebook includes code for this)
3. **Update the data path:**
   ```python
   data_path = '/content/drive/MyDrive/arXiv_scientific dataset.csv'
   ```
4. **Run all cells sequentially** (Runtime → Run all)

### Running Locally

1. **Modify the data loading section** in the notebook:
   ```python
   # Comment out Google Drive mounting
   # from google.colab import drive
   # drive.mount('/content/drive')
   
   # Update path to local file
   data_path = './arXiv_scientific dataset.csv'
   ```

2. **Execute the notebook cells in order:**
   - Part 1: Setup & Installation
   - Part 2: Load & Explore Data
   - Part 3: Data Sampling & Preprocessing
   - Part 4: Model Setup & Tokenization
   - Part 5: Training Configuration
   - Part 6: Model Training
   - Part 7: Model Evaluation
   - Part 8: Qualitative Analysis

### Expected Training Time
- **GPU (T4/V100):** 30-60 minutes
- **CPU:** 4-8 hours (not recommended)

##  Project Structure

```
Final/
├── 62FIT4ATI_Group27_Topic_SCIENTIFIC_PAPER_TITLE_GENERATION.ipynb  # Main notebook
├── README.md                    # This file
├── results/                     # Training checkpoints (created during training)
│   └── logs/                    # Training logs
├── final_model/                 # Saved fine-tuned model (created after training)
└── test_predictions.csv         # Evaluation results (created after evaluation)
```

##  Model Details

| Parameter | Value |
|-----------|-------|
| Base Model | `t5-base` |
| Parameters | ~220M |
| Max Input Length | 512 tokens |
| Max Target Length | 64 tokens |
| Task Prefix | `"summarize: "` |

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 3 |
| Batch Size | 8 |
| Gradient Accumulation Steps | 4 |
| Effective Batch Size | 32 |
| Learning Rate | 5e-5 |
| Warmup Steps | 500 |
| Weight Decay | 0.01 |
| FP16 | Enabled (if GPU available) |
| Beam Search | 4 beams |

##  Evaluation Metrics

The model is evaluated using:

- **ROUGE-1:** Unigram overlap between generated and reference titles
- **ROUGE-2:** Bigram overlap
- **ROUGE-L:** Longest common subsequence
- **BLEU:** Bilingual Evaluation Understudy score

### Qualitative Categories
Predictions are categorized based on similarity score:
- **High Match:** Similarity > 0.6
- **Medium Match:** Similarity 0.3-0.6
- **Low Match:** Similarity < 0.3

##  Usage

### Interactive Title Generation

After training, use the `generate_title()` function:

```python
# Generate a single title
titles = generate_title(
    "Your scientific abstract here...",
    num_beams=4,
    max_length=64,
    num_return_sequences=1
)
print(titles[0])

# Generate multiple title options
titles = generate_title(
    "Your scientific abstract here...",
    num_beams=5,
    num_return_sequences=3
)
for i, title in enumerate(titles):
    print(f"{i+1}. {title}")
```

### Example

```python
abstract = """
We present a novel approach to natural language processing using 
transformer-based architectures. Our method achieves state-of-the-art 
results on multiple benchmarks while being more computationally 
efficient than previous approaches.
"""

titles = generate_title(abstract, num_return_sequences=3)
# Output:
# 1. A Novel Transformer-Based Approach to Natural Language Processing
# 2. Efficient Transformer Architectures for NLP
# 3. State-of-the-Art NLP with Computational Efficiency
```

##  Authors

**Group 27 - 62FIT4ATI**

##  License

This project is for educational purposes as part of the ATI course.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [arXiv Dataset](https://arxiv.org/)
- T5 Model by Google Research


# If you want to use model, please download my model and push in your drive, then run code below(Recommend run on google coolab)
```python
# ============================================================
# PART 11: LOAD SAVED MODEL & INFERENCE 
# ============================================================

# 1. Install library
# !pip install -q transformers sentencepiece
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from google.colab import drive

# 2. Mount Google Drive 
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

model_path = "/content/drive/MyDrive/t5_title_generation_model"

print(f" Loading model from: {model_path}...")

# 4. Load Model & Tokenizer từ Drive
try:
    # Load tokenizer
    loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Load model
    loaded_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)
    
    print(" Model loaded successfully!")
    print(f"   Device: {device}")
    
except Exception as e:
    print(f" Error loading model: {e}")
    print("  Check again in your Google Drive.")

# 5. (Summarize Function)
def generate_title_from_saved_model(text, max_length=64, num_beams=4):
    """
    Use model
    """
    # Preprocess
    input_text = "summarize: " + text
    
    # Tokenize
    inputs = loaded_tokenizer(
        input_text, 
        max_length=512, 
        truncation=True, 
        return_tensors="pt"
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = loaded_model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            length_penalty=1.0
        )
    
    # Decode text
    title = loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return title

# ============================================================
# DEMO: TEST 
# ============================================================

print("\n Testing Inference...")

my_abstract = """
Deep learning models have achieved remarkable success in various natural language processing tasks. 
However, their performance often relies heavily on large-scale annotated datasets, which are expensive 
and time-consuming to acquire. In this paper, we propose a semi-supervised learning approach that 
leverages unlabeled data to improve model generalization. Our method utilizes a consistency regularization 
technique combined with a novel data augmentation strategy. Experimental results on three benchmark 
datasets demonstrate that our approach significantly outperforms existing baselines, achieving comparable 
results to fully supervised methods with only 10% of the labeled data.
"""

print("-" * 50)
print(" Input Abstract:")
print(my_abstract.strip())

generated_title = generate_title_from_saved_model(my_abstract)

print("\n Generated Title:")
print(f" {generated_title}")
print("-" * 50)
```
