import logging
import torch
import torch.nn as nn
from typing import List, Any, Optional
from transformers import AutoModel, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

from .classifier import DNABERT2Classifier

logger = logging.getLogger(__name__)

# --- Custom Bottleneck Adapter Module ---
class BottleneckAdapter(nn.Module):
    """
    A lightweight, custom bottleneck adapter block in PyTorch.
    Projects hidden state to bottleneck_dim, applies non-linearity and dropout,
    then projects back to hidden_size with a residual skip connection.
    """
    def __init__(
        self, 
        hidden_size: int, 
        bottleneck_dim: int = 64, 
        activation: str = "relu", 
        dropout: float = 0.1
    ):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_dim)
        if activation.lower() == "relu":
            self.act = nn.ReLU()
        elif activation.lower() == "gelu":
            self.act = nn.GELU()
        else:
            logger.warning(f"Unknown activation '{activation}', using ReLU.")
            self.act = nn.ReLU()
            
        self.up = nn.Linear(bottleneck_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights (standard bottleneck initialization: small weights near zero for skip bypass)
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(x)
        h = self.act(h)
        h = self.dropout(h)
        h = self.up(h)
        return x + h

class AdaptedLayerWrapper(nn.Module):
    """
    Wraps a transformer layer to insert a bottleneck adapter after its output.
    """
    def __init__(self, original_layer: nn.Module, adapter: BottleneckAdapter):
        super().__init__()
        self.original_layer = original_layer
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        outputs = self.original_layer(*args, **kwargs)
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            # Apply adapter to sequence representation hidden states
            hidden_states = self.adapter(hidden_states)
            return (hidden_states,) + outputs[1:]
        else:
            hidden_states = outputs
            hidden_states = self.adapter(hidden_states)
            return hidden_states


# --- Public PEFT Integrations ---
def get_peft_classifier(
    method: str,
    num_labels: int = 2,
    dropout: float = 0.1,
    pooling: str = "mean",
    model_name_or_path: str = "zhihan1996/DNABERT-2-117M",
    peft_config_dict: Optional[dict] = None
) -> DNABERT2Classifier:
    """
    Builds and returns a DNABERT2Classifier wrapped with the selected PEFT method:
    - 'full': Full fine-tuning baseline (all layers trainable).
    - 'lora': LoRA wrapped using HF peft.
    - 'qlora': Quantized 4-bit model wrapped with LoRA using peft and bitsandbytes.
    - 'adapters': Custom bottleneck adapter applied to transformer layers.
    """
    method = method.lower()
    peft_config_dict = peft_config_dict or {}
    
    if method == "full":
        # Load standard model
        encoder = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        classifier = DNABERT2Classifier(encoder, num_labels=num_labels, dropout=dropout, pooling=pooling)
        return classifier
        
    elif method == "lora":
        # Load standard encoder
        encoder = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        # Configure LoRA
        r = peft_config_dict.get("r", 8)
        alpha = peft_config_dict.get("alpha", 16)
        lora_dropout = peft_config_dict.get("dropout", 0.1)
        target_modules = peft_config_dict.get("target_modules", ["query", "value"])
        
        lora_config = LoraConfig(
            task_type=None,  # We wrap encoder backbone, classifier is trained normally
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none"
        )
        
        encoder = get_peft_model(encoder, lora_config)
        logger.info(f"LoRA wrapper applied to encoder. Target modules: {target_modules}, rank={r}")
        
        classifier = DNABERT2Classifier(encoder, num_labels=num_labels, dropout=dropout, pooling=pooling)
        return classifier
        
    elif method == "qlora":
        # Configure BitsAndBytes 4-bit double quantization
        # Dynamic check for CUDA support
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, which is required for 4-bit bitsandbytes quantization in QLoRA.")
            
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("bitsandbytes is not installed. Run 'pip install bitsandbytes' to run QLoRA.")
            
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        encoder = AutoModel.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        
        # Prepare model for k-bit training
        encoder = prepare_model_for_kbit_training(encoder, use_gradient_checkpointing=False)
        
        # Configure LoRA
        r = peft_config_dict.get("r", 8)
        alpha = peft_config_dict.get("alpha", 16)
        lora_dropout = peft_config_dict.get("dropout", 0.1)
        target_modules = peft_config_dict.get("target_modules", ["query", "value"])
        
        lora_config = LoraConfig(
            task_type=None,
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none"
        )
        
        encoder = get_peft_model(encoder, lora_config)
        logger.info(f"QLoRA 4-bit wrapper applied to encoder. Rank={r}")
        
        classifier = DNABERT2Classifier(encoder, num_labels=num_labels, dropout=dropout, pooling=pooling)
        return classifier
        
    elif method == "adapters":
        # Load standard model
        encoder = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        # Freeze base encoder parameters
        for param in encoder.parameters():
            param.requires_grad = False
            
        # Get hidden size
        hidden_size = encoder.config.hidden_size if hasattr(encoder.config, "hidden_size") else 768
        
        bottleneck_dim = peft_config_dict.get("bottleneck_dim", 64)
        activation = peft_config_dict.get("non_linearity", "relu")
        adapter_dropout = peft_config_dict.get("adapter_dropout", 0.1)
        
        # Find transformer layers (standard bert layers are inside encoder.layer)
        if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
            layers = encoder.encoder.layer
        elif hasattr(encoder, "layer"):
            layers = encoder.layer
        else:
            # Fallback if names are different in customized models
            layers = [m for m in encoder.modules() if m.__class__.__name__ in ("BertLayer", "FlashBertLayer", "TransformerBlock")]
            
        if not layers:
            raise RuntimeError("Could not locate transformer layers in the DNABERT-2 encoder for Adapter insertion.")
            
        logger.info(f"Found {len(layers)} transformer layers. Inserting bottleneck adapters (bottleneck_dim={bottleneck_dim}).")
        
        # Insert adapter in each layer
        for i, layer in enumerate(layers):
            adapter = BottleneckAdapter(
                hidden_size=hidden_size,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                dropout=adapter_dropout
            )
            # Wrap layer
            if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
                encoder.encoder.layer[i] = AdaptedLayerWrapper(layer, adapter)
            elif hasattr(encoder, "layer"):
                encoder.layer[i] = AdaptedLayerWrapper(layer, adapter)
                
        # Unfreeze all adapter modules
        adapter_params = 0
        for m in encoder.modules():
            if isinstance(m, BottleneckAdapter):
                for p in m.parameters():
                    p.requires_grad = True
                    adapter_params += p.numel()
                    
        logger.info(f"Custom Adapter insertion complete. Added {adapter_params} trainable adapter parameters.")
        
        # Build classifier (classification head parameters are created and will have requires_grad=True by default)
        classifier = DNABERT2Classifier(encoder, num_labels=num_labels, dropout=dropout, pooling=pooling)
        return classifier
        
    else:
        raise ValueError(f"Unknown PEFT method: {method}. Supported: 'full', 'lora', 'qlora', 'adapters'")
