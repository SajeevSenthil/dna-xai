import logging
import os
import torch
import torch.nn as nn
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def track_efficiency(model: nn.Module) -> Dict[str, Any]:
    """
    Computes parameter efficiency details for any PyTorch module.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": (trainable_params / total_params) * 100 if total_params > 0 else 0.0
    }

def save_comparison_report(
    experiments_data: List[Dict[str, Any]], 
    output_dir: str,
    filename: str = "performance_comparison.md"
) -> None:
    """
    Saves a markdown comparative study table of different fine-tuning approaches.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, filename)
    
    headers = [
        "Method", "Accuracy", "Precision", "Recall", "F1", 
        "AUROC", "Trainable Params", "Total Params", "Trainable %", "Training Time (s)", "Peak VRAM (MB)"
    ]
    
    markdown_lines = []
    markdown_lines.append("# Comparative Fine-Tuning Study Results")
    markdown_lines.append(f"Generated comparing different adaptation pipelines.\n")
    
    # Table header
    markdown_lines.append("| " + " | ".join(headers) + " |")
    markdown_lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    
    for row in experiments_data:
        method = str(row.get("method", "Unknown")).upper()
        acc = f"{row.get('accuracy', 0.0):.4f}"
        prec = f"{row.get('precision', 0.0):.4f}"
        rec = f"{row.get('recall', 0.0):.4f}"
        f1 = f"{row.get('f1', 0.0):.4f}"
        auroc = f"{row.get('auroc', 0.0):.4f}"
        trainable = f"{row.get('trainable_parameters', 0):,}"
        total = f"{row.get('total_parameters', 0):,}"
        pct = f"{row.get('trainable_percentage', 0.0):.3f}%"
        dur = f"{row.get('training_duration_sec', 0.0):.1f}s"
        
        gpu = row.get("peak_gpu_memory_mb", "N/A")
        gpu_str = f"{gpu:.1f} MB" if isinstance(gpu, (int, float)) and gpu > 0 else "N/A"
        
        row_fields = [method, acc, prec, rec, f1, auroc, trainable, total, pct, dur, gpu_str]
        markdown_lines.append("| " + " | ".join(row_fields) + " |")
        
    with open(report_path, "w") as f:
        f.write("\n".join(markdown_lines))
        
    logger.info(f"Comparative report written successfully to: {report_path}")
