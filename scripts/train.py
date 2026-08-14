import os
import argparse
import logging
import yaml
import torch
import pandas as pd
import json

from src.utils.logging import setup_logging
from src.utils.seed import set_seed
from src.models.dnabert import load_dnabert_base
from src.models.peft_models import get_peft_classifier
from src.data.dataset import create_data_loader
from src.training.train import Trainer

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train DNABERT-2 classification model using PEFT or Full fine-tuning.")
    parser.add_argument("--method", type=str, required=True, choices=["full", "lora", "qlora", "adapters"],
                        help="Fine-tuning method: full, lora, qlora, adapters")
    parser.add_argument("--task", type=str, required=True, choices=["promoter", "tf"],
                        help="Classification task: promoter, tf")
    parser.add_argument("--tf_subdir", type=str, default="0", choices=["0", "1", "2", "3", "4"],
                        help="TF subdirectory (0 to 4) when task is tf.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument("--batch_size", type=int, help="Override training batch size.")
    parser.add_argument("--epochs", type=int, help="Override training epochs.")
    parser.add_argument("--lr", type=float, help="Override learning rate.")
    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Apply command line overrides to config dict
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["learning_rate"] = args.lr
        
    config["method"] = args.method
    config["task"] = args.task

    # Set paths and names based on task
    if args.task == "promoter":
        processed_dir = config["data"]["promoter"]["processed_dir"]
        checkpoint_name = f"promoter_{args.method}.pt"
        log_name = f"train_promoter_{args.method}.log"
    else:
        processed_dir = os.path.join(config["data"]["tf"]["processed_dir"], args.tf_subdir)
        checkpoint_name = f"tf_{args.tf_subdir}_{args.method}.pt"
        log_name = f"train_tf_{args.tf_subdir}_{args.method}.log"

    # Setup logging
    log_file_path = os.path.join("logs", log_name)
    setup_logging(log_file=log_file_path)
    logger.info(f"Starting training run: Task={args.task.upper()}, Method={args.method.upper()}")
    
    # Set seed
    set_seed(config.get("seed", 42))

    # Detect device
    device_name = config.get("device", "cuda")
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using compute device: {device}")

    # Load processed splits
    train_path = os.path.join(processed_dir, "train.csv")
    val_path = os.path.join(processed_dir, "val.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Preprocessed splits not found in {processed_dir}. Run scripts/prepare_data.py first!"
        )
        
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    logger.info(f"Loaded train samples: {len(train_df)} | val samples: {len(val_df)}")

    # Load base model & tokenizer
    base_model_name = config["model"]["name_or_path"]
    # We load base to get tokenizer, then get_peft_classifier handles model configuration
    base_encoder, tokenizer = load_dnabert_base(base_model_name)
    # Release base encoder to save memory before get_peft_classifier instantiates the target architecture
    del base_encoder
    torch.cuda.empty_cache()

    # Create DataLoaders
    max_length = config["model"].get("max_length", 128)
    batch_size = config["training"].get("batch_size", 8)
    
    train_loader = create_data_loader(
        train_df, tokenizer, max_length=max_length, batch_size=batch_size, shuffle=True
    )
    val_loader = create_data_loader(
        val_df, tokenizer, max_length=max_length, batch_size=batch_size, shuffle=False
    )

    # Initialize model with selected PEFT wrapper
    peft_config = config.get("peft", {}).get(args.method, {})
    model = get_peft_classifier(
        method=args.method,
        num_labels=2,
        dropout=config["training"].get("dropout", 0.1),
        pooling=config["training"].get("pooling", "mean"),
        model_name_or_path=base_model_name,
        peft_config_dict=peft_config
    )
    
    model = model.to(device)

    # Initialize Trainer and start training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_name=checkpoint_name
    )
    
    efficiency_stats = trainer.fit()
    
    # Save training efficiency details
    stats_dir = os.path.join(config["training"].get("checkpoint_dir", "experiments"), args.method)
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, f"{args.task}_efficiency_stats.json" if args.task == "promoter" else f"tf_{args.tf_subdir}_efficiency_stats.json")
    
    with open(stats_path, "w") as f:
        json.dump(efficiency_stats, f, indent=4)
        
    logger.info(f"Training run completed. Efficiency statistics saved to: {stats_path}")

if __name__ == "__main__":
    main()
