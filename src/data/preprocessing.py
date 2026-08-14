import re
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

def validate_sequence(seq: str) -> bool:
    """
    Validates if a sequence contains only valid DNA nucleotides (A, C, G, T, N).
    Supports uppercase sequences.
    """
    if not isinstance(seq, str) or len(seq) == 0:
        return False
    # Standard DNA bases + ambiguous base N
    return bool(re.match(r"^[ACGTN]+$", seq))

def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Preprocesses the raw DNA sequence DataFrame:
    1. Upper-cases all sequences.
    2. Removes leading/trailing whitespaces.
    3. Validates DNA characters, dropping invalid ones.
    4. Removes duplicate sequences.
    5. Normalizes labels to integers (0 or 1).
    6. Computes and logs preprocessing statistics.
    """
    stats = {}
    initial_count = len(df)
    stats["initial_samples"] = initial_count
    
    if initial_count == 0:
        raise ValueError("Empty DataFrame provided for preprocessing.")
        
    if "sequence" not in df.columns or "label" not in df.columns:
        raise ValueError("DataFrame must contain 'sequence' and 'label' columns.")

    # Convert sequences to uppercase and strip whitespace
    df = df.copy()
    df["sequence"] = df["sequence"].astype(str).str.upper().str.strip()
    
    # Validate characters
    valid_mask = df["sequence"].apply(validate_sequence)
    invalid_count = (~valid_mask).sum()
    df = df[valid_mask]
    stats["invalid_sequences_removed"] = int(invalid_count)
    
    # Remove duplicate sequences
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["sequence"])
    dedup_removed = before_dedup - len(df)
    stats["duplicate_sequences_removed"] = int(dedup_removed)
    
    # Normalize labels to binary
    df["label"] = df["label"].astype(int)
    unique_labels = df["label"].unique()
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"Labels must be binary (0 or 1), found: {unique_labels}")
        
    final_count = len(df)
    stats["final_samples"] = final_count
    
    # Label distribution
    pos_count = int((df["label"] == 1).sum())
    neg_count = int((df["label"] == 0).sum())
    stats["positive_samples"] = pos_count
    stats["negative_samples"] = neg_count
    stats["class_ratio_pos_neg"] = pos_count / neg_count if neg_count > 0 else float("inf")
    
    # Sequence length stats
    lengths = df["sequence"].apply(len)
    stats["min_length"] = int(lengths.min()) if final_count > 0 else 0
    stats["max_length"] = int(lengths.max()) if final_count > 0 else 0
    stats["mean_length"] = float(lengths.mean()) if final_count > 0 else 0.0
    
    logger.info(f"Preprocessing completed. Initial: {initial_count}, Final: {final_count}, "
                f"Removed: {invalid_count + dedup_removed} (Invalid: {invalid_count}, Dupes: {dedup_removed})")
    logger.info(f"Class distribution - Positives: {pos_count}, Negatives: {neg_count}, Ratio: {stats['class_ratio_pos_neg']:.3f}")
    logger.info(f"Sequence length stats - Min: {stats['min_length']}, Max: {stats['max_length']}, Mean: {stats['mean_length']:.1f}")
    
    return df, stats
