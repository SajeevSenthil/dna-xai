import os
import argparse
import logging
import yaml
import torch
import pandas as pd
import json

from src.utils.logging import setup_logging
from src.models.dnabert import load_dnabert_base
from src.data.dataset import create_data_loader
from src.training.evaluate import evaluate_checkpoint

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DNABERT-2 checkpoint on the test split.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model checkpoint (.pt file).")
    parser.add_argument("--task", type=str, required=True, choices=["promoter", "tf"],
                        help="Classification task: promoter, tf")
    parser.add_argument("--tf_subdir", type=str, default="0", choices=["0", "1", "2", "3", "4"],
                        help="TF subdirectory (0 to 4) when task is tf.")
    args = parser.parse_args()

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint header to get config parameters
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found at: {args.checkpoint}")
    checkpoint_data = torch.load(args.checkpoint, map_location=torch.device("cpu"))
    config = checkpoint_data["config"]
    method = config["method"]
    
    # Setup logging
    log_name = f"eval_{args.task}_{method}.log"
    setup_logging(log_file=os.path.join("logs", log_name))
    logger.info(f"Starting evaluation of checkpoint: {args.checkpoint}")
    
    # Load dataset split
    if args.task == "promoter":
        processed_dir = config["data"]["promoter"]["processed_dir"]
        output_prefix = f"promoter_{method}_"
    else:
        processed_dir = os.path.join(config["data"]["tf"]["processed_dir"], args.tf_subdir)
        output_prefix = f"tf_{args.tf_subdir}_{method}_"
        
    test_path = os.path.join(processed_dir, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test split not found at: {test_path}")
        
    test_df = pd.read_csv(test_path)
    logger.info(f"Loaded test split samples: {len(test_df)}")

    # Load tokenizer
    base_model_name = config["model"]["name_or_path"]
    _, tokenizer = load_dnabert_base(base_model_name)

    # Create test Loader
    max_length = config["model"].get("max_length", 128)
    batch_size = config["training"].get("batch_size", 8)
    test_loader = create_data_loader(
        test_df, tokenizer, max_length=max_length, batch_size=batch_size, shuffle=False
    )

    # Output directories
    metrics_out_dir = "results/metrics"
    plots_out_dir = "results/plots"
    os.makedirs(metrics_out_dir, exist_ok=True)
    os.makedirs(plots_out_dir, exist_ok=True)

    # Evaluate
    metrics, predictions_data = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        test_loader=test_loader,
        device=device,
        output_dir=plots_out_dir,
        prefix=output_prefix
    )

    # Save metrics JSON to results/metrics/
    metrics_save_path = os.path.join(metrics_out_dir, f"{output_prefix}metrics.json")
    with open(metrics_save_path, "w") as f:
        # Include metadata inside save file
        metrics["method"] = method
        metrics["task"] = args.task
        metrics["tf_subdir"] = args.tf_subdir if args.task == "tf" else None
        json.dump(metrics, f, indent=4)
        
    logger.info(f"Metrics saved to: {metrics_save_path}")

    # Save predictions structure (useful for explainability checks)
    predictions_path = os.path.join(metrics_out_dir, f"{output_prefix}predictions.json")
    with open(predictions_path, "w") as f:
        json.dump(predictions_data, f, indent=4)
        
    logger.info(f"Prediction outputs saved to: {predictions_path}")
    logger.info("Evaluation pipeline execution complete.")

if __name__ == "__main__":
    main()
