"""Data loading utilities for VietFinQA."""
import pandas as pd
from pathlib import Path

def load_articles(data_dir="data/raw", source="merged"):
    """Load raw articles from CSV.
    
    Args:
        data_dir: Directory containing raw CSV files
        source: "merged" (default), "baodautu", or "vneconomy"
    
    Returns:
        DataFrame with article data
    """
    path = Path(data_dir) / f"{source}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)

def load_split(split="train", data_dir="data/training_data"):
    """Load a dataset split (train/val/test).
    
    Args:
        split: "train", "val", "test", or "gold_test"
        data_dir: Directory containing split CSVs
    
    Returns:
        DataFrame with QA data
    """
    path = Path(data_dir) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}")
    return pd.read_csv(path)

def get_statistics(df):
    """Compute basic statistics on a dataset."""
    stats = {
        "num_examples": len(df),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
    }
    if "question" in df.columns:
        stats["avg_question_length"] = df["question"].str.len().mean()
    if "answer" in df.columns:
        stats["avg_answer_length"] = df["answer"].str.len().mean()
    return stats
