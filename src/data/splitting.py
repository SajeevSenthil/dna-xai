import logging
import pandas as pd
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)

def prevent_leakage_and_split(
    df: pd.DataFrame, 
    val_split: float = 0.1, 
    test_split: float = 0.1, 
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Deterministic splitting of deduplicated sequence DataFrame.
    Guarantees no sequence overlap across train, validation, and test splits.
    
    Returns:
        train_df, val_df, test_df
    """
    # Double check for duplicate sequences to ensure no exact sequence leakage
    df = df.copy()
    duplicates_count = df.duplicated(subset=["sequence"]).sum()
    if duplicates_count > 0:
        logger.warning(f"Found {duplicates_count} duplicates in split dataset. Deduplicating now.")
        df = df.drop_duplicates(subset=["sequence"])

    # Shuffle the dataset using the fixed random seed
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(df))
    df = df.iloc[shuffled_indices].reset_index(drop=True)

    # Compute split boundaries
    n = len(df)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_test - n_val

    # Slice the dataframe
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].reset_index(drop=True)

    # Verify complete independence of sequences
    train_seqs = set(train_df["sequence"])
    val_seqs = set(val_df["sequence"])
    test_seqs = set(test_df["sequence"])

    overlap_train_val = train_seqs.intersection(val_seqs)
    overlap_train_test = train_seqs.intersection(test_seqs)
    overlap_val_test = val_seqs.intersection(test_seqs)

    if len(overlap_train_val) > 0 or len(overlap_train_test) > 0 or len(overlap_val_test) > 0:
        raise ValueError(
            f"Data leakage detected! Overlaps - Train/Val: {len(overlap_train_val)}, "
            f"Train/Test: {len(overlap_train_test)}, Val/Test: {len(overlap_val_test)}"
        )

    logger.info(f"Splits created successfully (seed={seed}):")
    logger.info(f"  Train:      {len(train_df)} samples ({len(train_df)/n*100:.1f}%)")
    logger.info(f"  Validation: {len(val_df)} samples ({len(val_df)/n*100:.1f}%)")
    logger.info(f"  Test:       {len(test_df)} samples ({len(test_df)/n*100:.1f}%)")

    return train_df, val_df, test_df
