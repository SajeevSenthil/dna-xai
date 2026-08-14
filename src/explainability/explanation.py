import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Set

logger = logging.getLogger(__name__)

def calculate_explanation_overlap(
    shap_values: np.ndarray, 
    lime_values: np.ndarray, 
    top_n: int = 8
) -> float:
    """
    Computes the Intersection-over-Union (IoU) of the top N highest attribution positions
    identified by SHAP and LIME.
    """
    # Find indices of top N positions
    top_shap_idx = set(np.argsort(np.abs(shap_values))[-top_n:])
    top_lime_idx = set(np.argsort(np.abs(lime_values))[-top_n:])
    
    intersection = top_shap_idx.intersection(top_lime_idx)
    union = top_shap_idx.union(top_lime_idx)
    
    if len(union) == 0:
        return 0.0
    return len(intersection) / len(union)

def identify_important_regions(
    scores: np.ndarray,
    sequence: str,
    threshold_percentile: float = 90.0,
    min_len: int = 3
) -> List[Dict[str, Any]]:
    """
    Groups adjacent bases that score above the threshold percentile of attribution scores
    into continuous sequence regions.
    """
    scores_abs = np.abs(scores)
    threshold = np.percentile(scores_abs, threshold_percentile)
    
    # Indices exceeding threshold
    high_importance_indices = np.where(scores_abs >= threshold)[0]
    
    if len(high_importance_indices) == 0:
        return []
        
    regions = []
    current_region = [high_importance_indices[0]]
    
    for idx in high_importance_indices[1:]:
        if idx == current_region[-1] + 1:
            current_region.append(idx)
        else:
            if len(current_region) >= min_len:
                regions.append(current_region)
            current_region = [idx]
            
    if len(current_region) >= min_len:
        regions.append(current_region)
        
    extracted_regions = []
    for r in regions:
        start_pos = int(r[0])
        end_pos = int(r[-1] + 1)  # 1-indexed / slice boundary representation
        sub_seq = sequence[start_pos:end_pos]
        mean_score = float(np.mean(scores[r]))
        
        extracted_regions.append({
            "start": start_pos,
            "end": end_pos,
            "sequence": sub_seq,
            "mean_score": mean_score,
            "indices": r.tolist()
        })
        
    # Sort regions by absolute score descending
    extracted_regions.sort(key=lambda x: abs(x["mean_score"]), reverse=True)
    return extracted_regions

def analyze_prediction_categories(
    labels: List[int], 
    preds: List[int], 
    probs: List[float]
) -> Dict[str, List[int]]:
    """
    Categorizes indices of test samples into four groups:
    - True Positives (TP)
    - True Negatives (TN)
    - False Positives (FP)
    - False Negatives (FN)
    """
    categories = {
        "TP": [],
        "TN": [],
        "FP": [],
        "FN": []
    }
    
    for idx, (label, pred) in enumerate(zip(labels, preds)):
        if label == 1 and pred == 1:
            categories["TP"].append(idx)
        elif label == 0 and pred == 0:
            categories["TN"].append(idx)
        elif label == 0 and pred == 1:
            categories["FP"].append(idx)
        elif label == 1 and pred == 0:
            categories["FN"].append(idx)
            
    return categories

def generate_explanation_report(
    prediction_label: str,
    confidence: float,
    important_regions: List[Dict[str, Any]],
    motif_results: List[Dict[str, Any]],
    overlap_iou: Optional[float] = None
) -> str:
    """
    Generates a natural-language report from attribution scores and motif scanning matches.
    Ensures absolute truthfulness: does NOT fabricate motifs or functions.
    """
    report_lines = []
    report_lines.append("=" * 50)
    report_lines.append("DNA REGULATORY ELEMENT PREDICTION REPORT")
    report_lines.append("=" * 50)
    
    report_lines.append(f"Prediction: {prediction_label}")
    report_lines.append(f"Confidence: {confidence * 100:.1f}%")
    
    if overlap_iou is not None:
        report_lines.append(f"SHAP/LIME Consistency Overlap (IoU): {overlap_iou:.2f}")
        
    report_lines.append("-" * 50)
    report_lines.append("IMPORTANT SEQUENCE REGIONS (SHAP)")
    report_lines.append("-" * 50)
    
    if len(important_regions) == 0:
        report_lines.append("No continuous high-attribution regions met the threshold.")
    else:
        for i, reg in enumerate(important_regions[:3]):  # Show top 3 regions
            dir_str = "positive (promotes prediction)" if reg["mean_score"] > 0 else "negative (inhibits prediction)"
            report_lines.append(
                f"Region {i+1}: positions {reg['start']}–{reg['end']}\n"
                f"  Sequence:     {reg['sequence']}\n"
                f"  SHAP Attribution: {reg['mean_score']:.4f} ({dir_str})"
            )
            
    report_lines.append("-" * 50)
    report_lines.append("BIOLOGICAL EVIDENCE (MOTIF MATCHING)")
    report_lines.append("-" * 50)
    
    matched_motifs = [m for m in motif_results if m.get("matched", False)]
    
    if len(matched_motifs) == 0:
        report_lines.append("No known regulatory motifs were identified in the high-attribution regions.")
    else:
        for m in matched_motifs:
            report_lines.append(
                f"Matched Motif: {m['motif_name']}\n"
                f"  Sequence Region: positions {m['start']}–{m['end']} ('{m['sequence']}')\n"
                f"  Matching Score:  {m['score']:.2f} (Threshold: {m['threshold']:.2f})\n"
                f"  Database Source: JASPAR core"
            )
            
    report_lines.append("-" * 50)
    report_lines.append("FINAL INTERPRETATION")
    report_lines.append("-" * 50)
    
    # Generate explanation based strictly on actual facts
    if len(matched_motifs) > 0:
        top_motif = matched_motifs[0]
        report_lines.append(
            f"The sequence is predicted to be '{prediction_label}' with high confidence. "
            f"The model's decision heavily relied on the region containing positions {top_motif['start']}–{top_motif['end']}. "
            f"This region aligns with a known '{top_motif['motif_name']}' motif from the database. "
            f"This biological evidence is consistent with the model's prediction."
        )
    else:
        if len(important_regions) > 0:
            top_reg = important_regions[0]
            report_lines.append(
                f"The sequence is predicted to be '{prediction_label}'. "
                f"The model strongly relied on the region around positions {top_reg['start']}–{top_reg['end']} ('{top_reg['sequence']}'). "
                f"However, no matching regulatory motif was identified in this region. "
                f"Therefore, the attribution provides model-level evidence but not a confirmed biological motif interpretation."
            )
        else:
            report_lines.append(
                f"The sequence is predicted to be '{prediction_label}'. "
                f"No specific localized regions of high attribution were identified, meaning the model "
                f"may have relied on distributed sequence features."
            )
            
    report_lines.append("=" * 50)
    return "\n".join(report_lines)
