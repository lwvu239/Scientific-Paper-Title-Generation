# Scientific Paper Title Generation

This project implements an abstractive text summarization system that generates compelling, concise titles for scientific papers based on their abstracts. It fine-tunes a pre-trained **T5 (Text-to-Text Transfer Transformer)** model using the **arXiv Scientific Dataset**.

The repository has been modularized into separate Python scripts for cleaner project management, local execution, and easy configuration.

---

## 📂 Project Structure

```text
Scientific-Paper-Title-Generation/
│
├── config.py                 # Configuration, hyperparameters, and path definitions
├── data_loader.py            # Data loading, length filtering, sampling, and tokenization
├── train.py                  # Model training pipeline, checkpoint saving, and loss plotting
├── eval.py                   # Model evaluation (ROUGE, BLEU) and prediction classification
├── inference.py              # Interactive title generator & CLI demo
├── utils.py                  # Helper functions (seed, device mapping, metrics, similarity score)
├── README.md                 # Project documentation
│
├── arXiv_scientific dataset.csv # Input dataset (place in this folder)
├── results/                  # Training history and logs (created during training)
│   ├── logs/
│   ├── training_history.csv
│   └── loss_curves.png       # Generated training vs validation loss curve
└── final_model/              # Fine-tuned model checkpoint (created after training)
```

---

## ✨ Features

- **Modular Design:** Separate scripts for preprocessing, training, evaluation, and inference.
- **Data Pipeline:** Filters out outliers (e.g. very short/long titles/abstracts) and structures data into train, validation, and test splits.
- **Fine-tuning Pipeline:** Built on Hugging Face's `Seq2SeqTrainer` utilizing mixed precision training (FP16) if GPU is available.
- **Evaluation Suite:** Computes ROUGE-1, ROUGE-2, ROUGE-L, and BLEU scores. It also performs qualitative analysis by categorizing predictions into:
  - *High Match:* Similarity ratio > 0.6
  - *Medium Match:* Similarity ratio 0.3 - 0.6
  - *Low Match:* Similarity ratio < 0.3
- **Interactive Interface:** Easy-to-use `TitleGenerator` class for loading the model and predicting titles on arbitrary abstracts.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Scientific-Paper-Title-Generation
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install transformers datasets evaluate rouge-score bert-score sacrebleu sentencepiece accelerate torch pandas numpy matplotlib seaborn tqdm
```

---

## 🚀 Execution Guide

### Step 1: Dataset Preparation
1. Download the **arXiv Scientific Dataset** (`arXiv_scientific dataset.csv`).
2. Place the CSV file in the root of this project directory.

### Step 2: Model Training
Run `train.py` to preprocess the dataset, train the model, save weights, and plot training loss curves:
```bash
python train.py
```
* **Output:** Saves the model weights to `final_model/` and training curves to `results/loss_curves.png`.

### Step 3: Evaluation
Run `eval.py` to evaluate the model on the test dataset:
```bash
python eval.py
```
* **Output:** Displays test metrics (ROUGE and BLEU) and saves predicted vs reference titles to `test_predictions.csv` with their qualitative matching score.

### Step 4: Run Inference
Test the title generator on a sample abstract using:
```bash
python inference.py
```

---

## ⚙️ Configuration (`config.py`)

You can modify configurations inside [config.py](file:///d:/Scientific-Paper-Title-Generation/config.py) directly:
* `MODEL_NAME`: Underlying pre-trained model (default: `t5-base`).
* `SAMPLE_SIZE`: Total number of records to sample (default: `30000`).
* `MAX_INPUT_LENGTH`: Maximum token length for abstracts (default: `512`).
* `MAX_TARGET_LENGTH`: Maximum token length for titles (default: `64`).
* `num_train_epochs`: Number of training iterations (default: `3`).
* `per_device_train_batch_size`: Batch size per device (default: `8`).
* `gradient_accumulation_steps`: Number of steps for gradient accumulation (default: `4`).

---

## 💡 Usage Example

Use the `TitleGenerator` class in your own scripts:

```python
from inference import TitleGenerator

# Initialize the title generator (loads model from final_model/ by default)
generator = TitleGenerator()

# Define abstract
abstract = """
Deep learning models have achieved remarkable success in various natural language processing tasks. 
However, their performance often relies heavily on large-scale annotated datasets, which are expensive 
and time-consuming to acquire. In this paper, we propose a semi-supervised learning approach that 
leverages unlabeled data to improve model generalization. Our method utilizes a consistency regularization 
technique combined with a novel data augmentation strategy. Experimental results on three benchmark 
datasets demonstrate that our approach significantly outperforms existing baselines, achieving comparable 
results to fully supervised methods with only 10% of the labeled data.
"""

# Generate titles (e.g. 3 candidate options using beam search)
titles = generator.generate(abstract, num_beams=5, num_return_sequences=3)

for i, title in enumerate(titles):
    print(f"Candidate {i+1}: {title}")
```

---

## 🧑‍💻 Author

**Luu Quang Vu**
* ATI Course - Educational Project.
