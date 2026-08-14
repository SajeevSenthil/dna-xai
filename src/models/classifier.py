import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DNABERT2Classifier(nn.Module):
    """
    Sequence Classification Model wrapper for DNABERT-2.
    Extracts contextual embeddings from DNABERT-2, performs pooling
    (mean or CLS), applies dropout, and feeds into a linear classification head.
    """
    def __init__(
        self, 
        encoder: nn.Module, 
        num_labels: int = 2, 
        dropout: float = 0.1, 
        pooling: str = "mean"
    ):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling.lower()
        
        # Verify hidden size from DNABERT-2 config (usually 768)
        if hasattr(encoder.config, "hidden_size"):
            self.hidden_size = encoder.config.hidden_size
        elif hasattr(encoder.config, "d_model"):
            self.hidden_size = encoder.config.d_model
        else:
            logger.warning("Could not determine hidden size from encoder configuration. Defaulting to 768.")
            self.hidden_size = 768
            
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        logger.info(f"Initialized DNABERT2Classifier with pooling={self.pooling}, "
                    f"dropout={dropout}, hidden_size={self.hidden_size}, num_labels={num_labels}")

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            input_ids: Tensor of token ids (batch_size, sequence_length)
            attention_mask: Mask to avoid performing attention on padding token indices (batch_size, sequence_length)
        Returns:
            logits: Output logits of shape (batch_size, num_labels)
        """
        # Forward pass through base DNABERT-2 encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # Extract sequence representation
        # Shape: (batch_size, sequence_length, hidden_size)
        last_hidden_state = outputs.last_hidden_state
        
        if self.pooling == "mean":
            # Masked average pooling
            if attention_mask is not None:
                # Expand attention mask to match embedding size
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                sum_mask = input_mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = torch.mean(last_hidden_state, 1)
        elif self.pooling == "cls":
            # Extract first token representation (typically [CLS])
            pooled = last_hidden_state[:, 0]
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling}")
            
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits
