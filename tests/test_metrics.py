import numpy as np
import pytest
from src.utils.metrics import compute_metrics

def test_compute_metrics_perfect():
    y_true = [1, 0, 1, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 1, 0]
    y_prob = [0.9, 0.1, 0.85, 0.05, 0.99, 0.12]
    
    res = compute_metrics(y_true, y_pred, y_prob)
    
    assert res["accuracy"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0
    assert res["auroc"] == 1.0

def test_compute_metrics_standard():
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 0, 1, 0, 0]
    y_prob = [0.90, 0.10, 0.80, 0.30, 0.20]
    
    # Positive samples: 3. Negatives: 2.
    # True Positives (TP): 2 (samples 0, 2)
    # True Negatives (TN): 2 (samples 1, 4)
    # False Positives (FP): 0
    # False Negatives (FN): 1 (sample 3)
    #
    # Accuracy = (2+2)/5 = 0.8
    # Precision = 2/(2+0) = 1.0
    # Recall = 2/(2+1) = 2/3 ~ 0.6667
    # F1 = 2 * (1.0 * (2/3)) / (1.0 + (2/3)) = 4/5 = 0.8
    
    res = compute_metrics(y_true, y_pred, y_prob)
    
    assert abs(res["accuracy"] - 0.8) < 1e-4
    assert abs(res["precision"] - 1.0) < 1e-4
    assert abs(res["recall"] - 0.66667) < 1e-4
    assert abs(res["f1"] - 0.8) < 1e-4
    assert res["auroc"] == 1.0  # Perfect ranking order of y_prob relative to y_true
