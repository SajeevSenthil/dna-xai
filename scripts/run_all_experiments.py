import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import logging
import subprocess
import json
import yaml

from src.utils.logging import setup_logging
from src.training.experiment import save_comparison_report

logger = logging.getLogger(__name__)

def run_command(cmd_list: list) -> bool:
    """
    Safely executes a shell command and logs output.
    """
    cmd_str = " ".join(cmd_list)
    logger.info(f"Running command: {cmd_str}")
    try:
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command '{cmd_str}':\n{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Failed to launch command '{cmd_str}': {e}")
        return False

def aggregate_results(task: str, tf_subdir: str = "0", config_path: str = "config.yaml") -> None:
    """
    Collects performance JSONs and parameter efficiency stats for all methods,
    and aggregates them into a comparison Markdown study table.
    """
    methods = ["full", "lora", "qlora", "adapters"]
    aggregated_data = []

    # Load configuration to read output directories
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    checkpoint_dir = config["training"].get("checkpoint_dir", "experiments")
    
    logger.info(f"Aggregating results for task: {task.upper()}...")

    for method in methods:
        output_prefix = f"promoter_{method}_" if task == "promoter" else f"tf_{tf_subdir}_{method}_"
        
        # Paths to output files
        metrics_path = os.path.join("results/metrics", f"{output_prefix}metrics.json")
        
        if task == "promoter":
            efficiency_path = os.path.join(checkpoint_dir, method, "promoter_efficiency_stats.json")
        else:
            efficiency_path = os.path.join(checkpoint_dir, method, f"tf_{tf_subdir}_efficiency_stats.json")
            
        row = {"method": method}
        found_data = False
        
        # Load test set accuracy, f1, precision, recall, auroc
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    metrics_data = json.load(f)
                row.update({
                    "accuracy": metrics_data.get("accuracy", 0.0),
                    "precision": metrics_data.get("precision", 0.0),
                    "recall": metrics_data.get("recall", 0.0),
                    "f1": metrics_data.get("f1", 0.0),
                    "auroc": metrics_data.get("auroc", 0.0)
                })
                found_data = True
            except Exception as e:
                logger.error(f"Error reading metrics for {method}: {e}")
                
        # Load training trainable params, GPU memory, duration
        if os.path.exists(efficiency_path):
            try:
                with open(efficiency_path, "r") as f:
                    eff_data = json.load(f)
                row.update({
                    "total_parameters": eff_data.get("total_parameters", 0),
                    "trainable_parameters": eff_data.get("trainable_parameters", 0),
                    "trainable_percentage": eff_data.get("trainable_percentage", 0.0),
                    "training_duration_sec": eff_data.get("training_duration_sec", 0.0),
                    "peak_gpu_memory_mb": eff_data.get("peak_gpu_memory_mb", "N/A")
                })
                found_data = True
            except Exception as e:
                logger.error(f"Error reading efficiency for {method}: {e}")
                
        if found_data:
            aggregated_data.append(row)
        else:
            logger.warning(f"No results found for method {method.upper()} at expected paths.")
            
    if len(aggregated_data) > 0:
        report_filename = f"comparison_{task}.md" if task == "promoter" else f"comparison_tf_{tf_subdir}.md"
        save_comparison_report(
            experiments_data=aggregated_data,
            output_dir="results/tables",
            filename=report_filename
        )
    else:
        logger.error("No experimental results found to aggregate!")

def main():
    parser = argparse.ArgumentParser(description="Run full DNA-XAI experimental pipeline.")
    parser.add_argument("--task", type=str, default="promoter", choices=["promoter", "tf"],
                        help="Task to run: promoter, tf")
    parser.add_argument("--tf_subdir", type=str, default="0", choices=["0", "1", "2", "3", "4"],
                        help="TF subdirectory (0 to 4) when task is tf.")
    parser.add_argument("--aggregate_only", action="store_true",
                        help="Only compile existing results JSONs without launching training runs.")
    parser.add_argument("--method", type=str, choices=["full", "lora", "qlora", "adapters"],
                        help="Optionally run training for a single specific method instead of all.")
    args = parser.parse_args()

    setup_logging(log_file="logs/run_all_experiments.log")

    if args.aggregate_only:
        aggregate_results(args.task, args.tf_subdir)
        return

    logger.info("Starting DNA-XAI Experiment Pipeline execution...")

    # 1. Prepare/Split data
    logger.info("Step 1: Preparing datasets...")
    success = run_command(["python", "scripts/prepare_data.py"])
    if not success:
        logger.error("Data preparation failed! Terminating pipeline.")
        return

    # Determine which methods to run
    methods_to_run = [args.method] if args.method else ["full", "lora", "qlora", "adapters"]

    # 2. Run training and evaluation for each method
    for method in methods_to_run:
        logger.info(f"\nRunning training for {method.upper()}...")
        train_cmd = [
            "python", "scripts/train.py",
            "--method", method,
            "--task", args.task
        ]
        if args.task == "tf":
            train_cmd.extend(["--tf_subdir", args.tf_subdir])
            
        success = run_command(train_cmd)
        if not success:
            logger.warning(f"Training failed for method {method.upper()}. Skipping evaluation for this method.")
            continue

        # Evaluate checkpoint
        logger.info(f"Evaluating {method.upper()} checkpoint...")
        checkpoint_name = f"promoter_{method}.pt" if args.task == "promoter" else f"tf_{args.tf_subdir}_{method}.pt"
        checkpoint_path = os.path.join("experiments", checkpoint_name)
        
        eval_cmd = [
            "python", "scripts/evaluate.py",
            "--checkpoint", checkpoint_path,
            "--task", args.task
        ]
        if args.task == "tf":
            eval_cmd.extend(["--tf_subdir", args.tf_subdir])
            
        run_command(eval_cmd)

    # 3. Aggregate all results into comparison study tables
    logger.info("\nStep 3: Aggregating study outcomes...")
    aggregate_results(args.task, args.tf_subdir)
    logger.info("Pipeline execution finished.")

if __name__ == "__main__":
    main()
