import pandas as pd
import pytest
from src.data.preprocessing import validate_sequence, preprocess_dataframe
from src.data.splitting import prevent_leakage_and_split

def test_validate_sequence():
    # Only uppercase valid characters (A, C, G, T, N) should pass
    assert validate_sequence("ATCGN") is True
    assert validate_sequence("AAAA") is True
    assert validate_sequence("ATCGX") is False
    assert validate_sequence("atcgn") is False  # validates pre-uppercased strings
    assert validate_sequence("") is False
    assert validate_sequence(123) is False

def test_preprocess_dataframe():
    # Test valid data preprocessing
    raw_data = pd.DataFrame({
        "sequence": ["ATCG", "atcg", "ATCGX", "NNNN", "AATTGGCC"],
        "label": [1, 1, 0, 0, 1]
    })
    
    cleaned_df, stats = preprocess_dataframe(raw_data)
    
    # ATCGX is invalid (removed)
    # atcg is duplicate (removed)
    # Remaining should be ATCG, NNNN, AATTGGCC (len = 3)
    assert len(cleaned_df) == 3
    assert stats["initial_samples"] == 5
    assert stats["final_samples"] == 3
    assert stats["invalid_sequences_removed"] == 1
    assert stats["duplicate_sequences_removed"] == 1
    assert "ATCG" in cleaned_df["sequence"].values
    assert "NNNN" in cleaned_df["sequence"].values

def test_prevent_leakage_and_split():
    # Test deterministic split ratios and sequence independence
    df = pd.DataFrame({
        "sequence": [f"ATCG{i}" for i in range(100)],
        "label": [i % 2 for i in range(100)]
    })
    
    train1, val1, test1 = prevent_leakage_and_split(df, val_split=0.10, test_split=0.10, seed=42)
    train2, val2, test2 = prevent_leakage_and_split(df, val_split=0.10, test_split=0.10, seed=42)
    
    # Test reproducibility
    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(val1, val2)
    pd.testing.assert_frame_equal(test1, test2)
    
    # Test ratios
    assert len(train1) == 80
    assert len(val1) == 10
    assert len(test1) == 10
    
    # Test no overlapping sequences across splits
    train_seqs = set(train1["sequence"])
    val_seqs = set(val1["sequence"])
    test_seqs = set(test1["sequence"])
    
    assert len(train_seqs.intersection(val_seqs)) == 0
    assert len(train_seqs.intersection(test_seqs)) == 0
    assert len(val_seqs.intersection(test_seqs)) == 0
