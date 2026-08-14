import numpy as np
import pytest
from src.explainability.explanation import (
    calculate_explanation_overlap, 
    identify_important_regions, 
    generate_explanation_report
)
from src.motifs.motif_analysis import scan_sequence_for_motifs

def test_calculate_explanation_overlap():
    shap_vals = np.array([0.05, 0.80, 0.01, 0.95, 0.10])
    lime_vals = np.array([0.01, 0.75, 0.05, 0.90, 0.02])
    
    # Top 2 shap indices: 1 (0.80) and 3 (0.95)
    # Top 2 lime indices: 1 (0.75) and 3 (0.90)
    # Intersection = {1, 3}, Union = {1, 3} -> IoU = 1.0
    iou = calculate_explanation_overlap(shap_vals, lime_vals, top_n=2)
    assert iou == 1.0
    
    # Disjoint top indices
    shap_disjoint = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    lime_disjoint = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    iou_disjoint = calculate_explanation_overlap(shap_disjoint, lime_disjoint, top_n=1)
    assert iou_disjoint == 0.0

def test_identify_important_regions():
    scores = np.array([0.1, 0.1, 0.95, 0.98, 0.92, 0.1, 0.1, 0.1])
    sequence = "ATCGNATA"
    
    # 60th percentile threshold will select the highest scores: 0.98, 0.95, 0.92 (indices 2, 3, 4)
    regions = identify_important_regions(scores, sequence, threshold_percentile=60, min_len=3)
    
    assert len(regions) == 1
    assert regions[0]["start"] == 2
    assert regions[0]["end"] == 5
    assert regions[0]["sequence"] == "CGN"
    assert abs(regions[0]["mean_score"] - 0.95) < 0.05

def test_scan_sequence_for_motifs():
    # TATA consensus core is TATAAAAA, search for it
    seq_tata = "GCTATAAAAAGCGT"
    matches = scan_sequence_for_motifs(seq_tata, threshold=0.75)
    
    assert len(matches) > 0
    tata_matches = [m for m in matches if m["motif_name"] == "TATA-box"]
    assert len(tata_matches) > 0
    assert tata_matches[0]["start"] == 2


def test_generate_explanation_report_grounded():
    prediction_label = "Promoter"
    confidence = 0.95
    important_regions = [{
        "start": 5,
        "end": 12,
        "sequence": "TATAAAA",
        "mean_score": 0.42
    }]
    # Case 1: with matching motif
    motif_results_matched = [{
        "matched": True,
        "motif_name": "TATA-box",
        "start": 5,
        "end": 12,
        "sequence": "TATAAAA",
        "score": 0.95,
        "threshold": 0.75
    }]
    
    report_matched = generate_explanation_report(
        prediction_label, confidence, important_regions, motif_results_matched, overlap_iou=0.80
    )
    
    assert "Prediction: Promoter" in report_matched
    assert "Confidence: 95.0%" in report_matched
    assert "JASPAR core" in report_matched
    assert "TATAAAA" in report_matched
    assert "TATA-box" in report_matched
    
    # Case 2: without matching motif
    report_unmatched = generate_explanation_report(
        prediction_label, confidence, important_regions, [], overlap_iou=0.80
    )
    
    assert "No known regulatory motifs were identified" in report_unmatched
    assert "attribution provides model-level evidence but not a confirmed biological motif" in report_unmatched
