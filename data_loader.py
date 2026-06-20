import pandas as pd
from datasets import Dataset, DatasetDict
import config

def load_and_preprocess_data():
    """
    Loads raw CSV data, cleans missing values, filters length outliers,
    samples the dataset, and returns Hugging Face DatasetDict along with the raw test DataFrame.
    """
    DATA_PATH = 'arXiv_scientific dataset.csv'
    print(f"Loading dataset from: {config.DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {len(df)} samples")
    
    summary_col = 'summary'
    title_col = 'title'
    
    # Check for column existence
    if summary_col not in df.columns or title_col not in df.columns:
        # Fallback to search columns that look like 'summary' or 'title'
        cols = df.columns.tolist()
        summary_candidates = [c for c in cols if 'summary' in c.lower() or 'abstract' in c.lower()]
        title_candidates = [c for c in cols if 'title' in c.lower()]
        if summary_candidates:
            summary_col = summary_candidates[0]
        if title_candidates:
            title_col = title_candidates[0]
            
    print(f"Using summary column: '{summary_col}' and title column: '{title_col}'")
    
    # Remove missing values
    df_clean = df[[summary_col, title_col]].dropna()
    print(f"After removing missing values: {len(df_clean)} samples")
    
    # Calculate word lengths
    df_clean['summary_length'] = df_clean[summary_col].str.split().str.len()
    df_clean['title_length'] = df_clean[title_col].str.split().str.len()
    
    # Random Sampling
    if len(df_clean) > config.SAMPLE_SIZE:
        df_sample = df_clean.sample(n=config.SAMPLE_SIZE, random_state=config.SEED).reset_index(drop=True)
    else:
        df_sample = df_clean.reset_index(drop=True)
        
    print(f"Sampled dataset: {len(df_sample)} samples")
    
    # Filter by text lengths
    df_filtered = df_sample[
        (df_sample['summary_length'] >= config.MIN_SUMMARY_LENGTH) &
        (df_sample['summary_length'] <= config.MAX_SUMMARY_LENGTH) &
        (df_sample['title_length'] >= config.MIN_TITLE_LENGTH) &
        (df_sample['title_length'] <= config.MAX_TITLE_LENGTH)
    ].reset_index(drop=True)
    
    print(f"After filtering length outliers: {len(df_filtered)} samples")
    
    # Split data: 80% train, 10% validation, 10% test
    train_size = int(0.8 * len(df_filtered))
    val_size = int(0.1 * len(df_filtered))
    
    train_df = df_filtered[:train_size]
    val_df = df_filtered[train_size:train_size+val_size]
    test_df = df_filtered[train_size+val_size:]
    
    print(f"Data Split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    def create_hf_dataset(df):
        return Dataset.from_dict({
            'summary': df[summary_col].tolist(),
            'title': df[title_col].tolist()
        })
        
    dataset_dict = DatasetDict({
        'train': create_hf_dataset(train_df),
        'validation': create_hf_dataset(val_df),
        'test': create_hf_dataset(test_df)
    })
    
    return dataset_dict, test_df

def tokenize_datasets(dataset_dict, tokenizer):
    """
    Applies tokenization to train/val/test splits using T5 summary task prefix.
    """
    def preprocess_function(examples):
        # T5 requires a prefix for the task
        inputs = ["summarize: " + doc for doc in examples['summary']]
        model_inputs = tokenizer(
            inputs,
            max_length=config.MAX_INPUT_LENGTH,
            truncation=True,
            padding=False  # Pad dynamically during training
        )
        
        # Tokenize targets
        labels = tokenizer(
            examples['title'],
            max_length=config.MAX_TARGET_LENGTH,
            truncation=True,
            padding=False
        )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    print("Tokenizing datasets...")
    tokenized_datasets = dataset_dict.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset_dict['train'].column_names,
        desc="Tokenizing"
    )
    return tokenized_datasets
