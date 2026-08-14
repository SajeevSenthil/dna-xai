import logging
import os
import json
import time
import torch
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader

from src.models.peft_models import get_peft_classifier
from src.utils.metrics import compute_metrics, plot_confusion_matrix, plot_roc_pr_curves

logger = logging.getLogger(__name__)

def evaluate_checkpoint(
    checkpoint_path: str,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    prefix: str = ""
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Loads a saved model checkpoint, runs evaluation on the provided test loader,
    saves metrics as JSON, and plots evaluation curves.
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint data
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    epoch = checkpoint["epoch"]
    
    # Extract config parameters
    method = config["method"]
    num_labels = 2
    dropout = config["training"].get("dropout", 0.1)
    pooling = config["training"].get("pooling", "mean")
    model_name = config["model"]["name_or_path"]
    peft_config = config.get("peft", {}).get(method, {})
    
    # Re-initialize the classifier structure
    model = get_peft_classifier(
        method=method,
        num_labels=num_labels,
        dropout=dropout,
        pooling=pooling,
        model_name_or_path=model_name,
        peft_config_dict=peft_config
    )
    
    # Load weight state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded {method.upper()} model trained for {epoch + 1} epochs.")
    
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    # Inference latency tracking
    start_time = time.time()
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # Softmax to get positive class probabilities
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    latency_sec = time.time() - start_time
    avg_inference_latency_ms = (latency_sec / len(test_loader.dataset)) * 1000
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = total_loss / len(test_loader)
    metrics["avg_inference_latency_ms"] = avg_inference_latency_ms
    
    logger.info(f"Evaluation complete for {method.upper()}:")
    logger.info(f"  Loss:      {metrics['loss']:.4f}")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1:        {metrics['f1']:.4f}")
    logger.info(f"  AUROC:     {metrics['auroc']:.4f}")
    logger.info(f"  Inference Latency: {avg_inference_latency_ms:.2f} ms/sample")

    # Save metrics JSON
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{prefix}metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to: {json_path}")
    
    # Generate and save plots
    plot_confusion_matrix(
        all_labels, 
        all_preds, 
        save_path=os.path.join(output_dir, f"{prefix}confusion_matrix.png"),
        title=f"Confusion Matrix ({method.upper()})"
    )
    
    plot_roc_pr_curves(
        all_labels, 
        all_probs, 
        save_dir=output_dir,
        prefix=prefix
    )
    logger.info(f"Evaluation plots saved to: {output_dir}")
    
    predictions_data = {
        "labels": [int(l) for l in all_labels],
        "preds": [int(p) for p in all_preds],
        "probs": [float(prob) for prob in all_probs]
    }
    
    return metrics, predictions_data
