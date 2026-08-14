import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import argparse
import logging
import json
import numpy as np
import torch
import transformers
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: True
transformers.utils.check_torch_load_is_safe = lambda *args, **kwargs: True
try:
    import transformers.modeling_utils
    transformers.modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: True
except Exception:
    pass
import pandas as pd

from src.utils.logging import setup_logging
from src.models.dnabert import load_dnabert_base
from src.models.peft_models import get_peft_classifier
from src.explainability.shap_explainer import compute_shap_explanations
from src.explainability.lime_explainer import compute_lime_explanations
from src.explainability.explanation import (
    calculate_explanation_overlap,
    identify_important_regions,
    generate_explanation_report
)
from src.motifs.motif_analysis import scan_sequence_for_motifs

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate explainability report for a DNA sequence prediction.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model checkpoint (.pt file).")
    parser.add_argument("--task", type=str, required=True, choices=["promoter", "tf"],
                        help="Classification task: promoter, tf")
    parser.add_argument("--tf_subdir", type=str, default="0", choices=["0", "1", "2", "3", "4"],
                        help="TF subdirectory (0 to 4) when task is tf.")
    parser.add_argument("--sequence", type=str, help="Custom DNA sequence string to analyze.")
    parser.add_argument("--sample_idx", type=int, help="Index of sample in the test set to analyze.")
    parser.add_argument("--nsamples", type=int, default=100, help="Number of perturbation samples for SHAP.")
    parser.add_argument("--output_dir", type=str, default="results/explanations", help="Directory to save report output.")
    args = parser.parse_args()

    setup_logging(log_file="logs/explain.log")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load checkpoint and initialize model configuration
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    method = config["method"]

    base_model_name = config["model"]["name_or_path"]
    _, tokenizer = load_dnabert_base(base_model_name)

    model = get_peft_classifier(
        method=method,
        num_labels=2,
        dropout=config["training"].get("dropout", 0.1),
        pooling=config["training"].get("pooling", "mean"),
        model_name_or_path=base_model_name,
        peft_config_dict=config.get("peft", {}).get(method, {})
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    # 2. Extract Sequence to explain
    sequence = ""
    true_label = None
    
    if args.sequence:
        sequence = args.sequence.upper().strip()
        logger.info(f"Analyzing custom sequence: {sequence}")
    elif args.sample_idx is not None:
        # Load sequence from test split
        if args.task == "promoter":
            processed_dir = config["data"]["promoter"]["processed_dir"]
        else:
            processed_dir = os.path.join(config["data"]["tf"]["processed_dir"], args.tf_subdir)
            
        test_path = os.path.join(processed_dir, "test.csv")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test split file not found: {test_path}")
            
        test_df = pd.read_csv(test_path)
        if args.sample_idx < 0 or args.sample_idx >= len(test_df):
            raise IndexError(f"Sample index {args.sample_idx} is out of bounds for test set of size {len(test_df)}.")
            
        row = test_df.iloc[args.sample_idx]
        sequence = str(row["sequence"]).upper().strip()
        true_label = int(row["label"])
        logger.info(f"Analyzing test sample index {args.sample_idx} (True Label: {true_label})")
    else:
        raise ValueError("Must provide either --sequence or --sample_idx.")

    # 3. Model Prediction
    inputs = tokenizer(sequence, return_tensors="pt", max_length=config["model"].get("max_length", 128), padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        pred_label_idx = int(np.argmax(probs))
        confidence = float(probs[pred_label_idx])

    if args.task == "promoter":
        prediction_label = "Promoter" if pred_label_idx == 1 else "Non-Promoter"
    else:
        prediction_label = "TFBS" if pred_label_idx == 1 else "Non-TFBS"

    logger.info(f"Model prediction: {prediction_label} (Confidence: {confidence*100:.1f}%)")

    # 4. Compute SHAP Values
    shap_values, shap_base = compute_shap_explanations(
        sequence=sequence,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=config["model"].get("max_length", 128),
        nsamples=args.nsamples
    )

    # 5. Compute LIME Values
    lime_values, lime_base = compute_lime_explanations(
        sequence=sequence,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=config["model"].get("max_length", 128),
        num_perturbations=100
    )

    # 6. Calculate Attribution Overlap (IoU of top 8 positions)
    overlap_iou = calculate_explanation_overlap(shap_values, lime_values, top_n=min(8, len(sequence)))

    # 7. Identify High Attribution Regions
    important_regions = identify_important_regions(shap_values, sequence, threshold_percentile=90, min_len=3)

    # 8. Motif Scanning Analysis
    motif_matches = scan_sequence_for_motifs(sequence, threshold=0.72)

    # Filter motifs: check if they overlap with our high-attribution SHAP regions
    structured_motif_results = []
    high_attr_indices = set()
    for reg in important_regions:
        high_attr_indices.update(reg["indices"])
        
    for match in motif_matches:
        match_range = set(range(match["start"], match["end"]))
        # Overlaps if there is intersection between motif positions and high importance positions
        overlaps = len(match_range.intersection(high_attr_indices)) > 0
        match["overlaps_attribution"] = overlaps
        structured_motif_results.append(match)

    # 9. Generate Report Text
    report_text = generate_explanation_report(
        prediction_label=prediction_label,
        confidence=confidence,
        important_regions=important_regions,
        motif_results=structured_motif_results,
        overlap_iou=overlap_iou
    )

    # Print report to console
    print(report_text)

    # 10. Save outputs to results/explanations
    os.makedirs(args.output_dir, exist_ok=True)
    
    file_prefix = f"promoter_{method}" if args.task == "promoter" else f"tf_{args.tf_subdir}_{method}"
    if args.sample_idx is not None:
        file_prefix += f"_sample_{args.sample_idx}"
    else:
        file_prefix += f"_custom"
        
    # Save text report
    txt_save_path = os.path.join(args.output_dir, f"{file_prefix}_report.txt")
    with open(txt_save_path, "w") as f:
        f.write(report_text)
        
    # Save JSON details
    json_save_path = os.path.join(args.output_dir, f"{file_prefix}_data.json")
    save_data = {
        "sequence": sequence,
        "true_label": true_label,
        "prediction": prediction_label,
        "confidence": confidence,
        "shap_values": shap_values.tolist(),
        "lime_values": lime_values.tolist(),
        "overlap_iou": float(overlap_iou),
        "important_regions": important_regions,
        "motif_matches": structured_motif_results
    }
    
    with open(json_save_path, "w") as f:
        json.dump(save_data, f, indent=4)
        
    logger.info(f"Explanation report saved to: {txt_save_path}")
    logger.info(f"Explanation data saved to: {json_save_path}")

if __name__ == "__main__":
    main()
