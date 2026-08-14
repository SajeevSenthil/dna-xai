import logging
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Any, Optional

from src.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)

class Trainer:
    """
    Standard Custom PyTorch Trainer for DNA sequence classification.
    Supports:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Early stopping
    - Trainable parameter filtering
    - Real-time logging of metrics and GPU memory
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        checkpoint_name: str = "best_model.pt"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_name = checkpoint_name
        
        # Hyperparameters
        self.epochs = config["training"].get("epochs", 5)
        self.lr = float(config["training"].get("learning_rate", 2e-5))
        self.weight_decay = float(config["training"].get("weight_decay", 0.01))
        self.grad_accum_steps = config["training"].get("gradient_accumulation_steps", 1)
        self.fp16 = config["training"].get("fp16", True) and (device.type == "cuda")
        self.patience = config["training"].get("early_stopping_patience", 3)
        self.checkpoint_dir = config["training"].get("checkpoint_dir", "experiments")
        
        # Filter trainable parameters (crucial for PEFT models!)
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.total_param_count = sum(p.numel() for p in self.model.parameters())
        self.trainable_param_count = sum(p.numel() for p in self.trainable_params)
        
        logger.info(f"Trainer initialization:")
        logger.info(f"  Total parameters:     {self.total_param_count:,}")
        logger.info(f"  Trainable parameters: {self.trainable_param_count:,} ({self.trainable_param_count/self.total_param_count*100:.3f}%)")
        
        # Set up loss function & optimizer
        # CrossEntropyLoss expects target shape (batch_size) containing class indices (0 or 1)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.trainable_params, 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # Mixed precision setup
        self.scaler = torch.cuda.amp.GradScaler() if self.fp16 else None
        
        # Track training metrics
        self.best_val_f1 = -1.0
        self.best_epoch = -1
        self.epochs_no_improve = 0

    def train_epoch(self, epoch: int) -> float:
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Train]")
        
        for step, batch in enumerate(progress_bar):
            # Move inputs to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass with mixed precision
            if self.fp16:
                with torch.cuda.amp.autocast():
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = self.criterion(logits, labels)
                # Scale loss for gradient accumulation
                loss = loss / self.grad_accum_steps
                self.scaler.scale(loss).backward()
            else:
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(logits, labels)
                loss = loss / self.grad_accum_steps
                loss.backward()
                
            total_loss += loss.item() * self.grad_accum_steps
            
            # Optimizer step (respecting gradient accumulation)
            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                if self.fp16:
                    self.scaler.unscale_(self.optimizer)
                    # Gradient clipping to stabilize training
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=1.0)
                    self.optimizer.step()
                    
                self.optimizer.zero_grad()
                
            progress_bar.set_postfix({"loss": f"{loss.item() * self.grad_accum_steps:.4f}"})
            
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Runs evaluation on a specific DataLoader.
        """
        self.model.eval()
        total_loss = 0.0
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            
            # Extract probabilities and predicted labels
            probs = torch.softmax(logits, dim=-1)[:, 1]  # Probabilities of class 1
            preds = torch.argmax(logits, dim=-1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
        metrics = compute_metrics(all_labels, all_preds, all_probs)
        metrics["loss"] = total_loss / len(loader)
        metrics["labels"] = all_labels
        metrics["preds"] = all_preds
        metrics["probs"] = all_probs
        return metrics

    def fit(self) -> Dict[str, Any]:
        """
        Runs the full training loop with validation and early stopping.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        start_time = time.time()
        
        # Monitor GPU memory if CUDA is used
        peak_gpu_mem = 0.0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            
        logger.info("Starting training loop...")
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.evaluate(self.val_loader)
            
            epoch_duration = time.time() - epoch_start
            
            # Read peak GPU memory
            if self.device.type == "cuda":
                # Convert bytes to megabytes
                current_peak = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
                peak_gpu_mem = max(peak_gpu_mem, current_peak)
                gpu_msg = f" | Peak VRAM: {current_peak:.1f} MB"
            else:
                gpu_msg = ""
                
            logger.info(
                f"Epoch {epoch+1:02d} completed in {epoch_duration:.1f}s | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | Val AUROC: {val_metrics['auroc']:.4f}{gpu_msg}"
            )
            
            # Checkpoint based on Validation F1 (highly robust for classification tasks)
            val_f1 = val_metrics["f1"]
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                
                # Save best checkpoint
                checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)
                # Clean saving state dict (handling PEFT models correctly)
                save_data = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_metrics": {k: val_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "auroc", "loss"]},
                    "config": self.config
                }
                torch.save(save_data, checkpoint_path)
                logger.info(f"  => Saved new best model to {checkpoint_path} (Val F1: {val_f1:.4f})")
            else:
                self.epochs_no_improve += 1
                logger.info(f"  => Validation F1 did not improve. (Patience: {self.epochs_no_improve}/{self.patience})")
                
            # Early stopping check
            if self.epochs_no_improve >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}!")
                break
                
        training_duration = time.time() - start_time
        logger.info(f"Training finished in {training_duration:.1f}s. Best Epoch: {self.best_epoch + 1} (Val F1: {self.best_val_f1:.4f})")
        
        return {
            "best_epoch": self.best_epoch,
            "best_val_f1": self.best_val_f1,
            "training_duration_sec": training_duration,
            "peak_gpu_memory_mb": peak_gpu_mem,
            "total_parameters": self.total_param_count,
            "trainable_parameters": self.trainable_param_count,
            "trainable_percentage": (self.trainable_param_count / self.total_param_count) * 100
        }
