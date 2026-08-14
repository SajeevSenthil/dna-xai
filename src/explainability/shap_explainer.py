import logging
import torch
import numpy as np
import shap
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class DNAShapWrapper:
    """
    Wraps the classifier and tokenizer to compute SHAP attributions at the base-pair level.
    Perturbs input sequences by masking selected base positions with 'N'.
    """
    def __init__(
        self, 
        model: torch.nn.Module, 
        tokenizer: Any, 
        device: torch.device, 
        sequence: str,
        max_length: int = 128
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.sequence_list = list(sequence)
        self.max_length = max_length

    def predict_mask_combinations(self, binary_masks: np.ndarray) -> np.ndarray:
        """
        Takes a matrix of binary masks (shape: num_combinations, sequence_length),
        generates perturbed DNA sequences (with 'N' at positions where mask is 0),
        and returns the model's prediction probability for the positive class (class 1).
        """
        perturbed_sequences = []
        for mask in binary_masks:
            perturbed_seq = "".join(
                [char if m == 1 else "N" for char, m in zip(self.sequence_list, mask)]
            )
            perturbed_sequences.append(perturbed_seq)
            
        # Run inference in batches to prevent GPU memory spikes
        self.model.eval()
        predictions = []
        batch_size = 32
        
        for i in range(0, len(perturbed_sequences), batch_size):
            batch_seqs = perturbed_sequences[i : i + batch_size]
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_seqs,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                logits = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                predictions.extend(probs)
                
        return np.array(predictions)

def compute_shap_explanations(
    sequence: str,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    max_length: int = 128,
    nsamples: int = 100
) -> Tuple[np.ndarray, float]:
    """
    Computes SHAP attribution values for each base in the DNA sequence.
    
    Returns:
        shap_values: Array of attribution scores of length equal to the sequence.
        base_value: The baseline prediction probability (when all positions are masked).
    """
    seq_len = len(sequence)
    wrapper = DNAShapWrapper(model, tokenizer, device, sequence, max_length)
    
    # Define reference (all bases masked to 'N')
    background = np.zeros((1, seq_len))
    
    # Initialize KernelExplainer
    explainer = shap.KernelExplainer(wrapper.predict_mask_combinations, background)
    
    # Explain the instance where all bases are present (represented by a mask of all 1s)
    instance_to_explain = np.ones((1, seq_len))
    
    logger.info(f"Running SHAP KernelExplainer on sequence of length {seq_len} (nsamples={nsamples})...")
    shap_vals = explainer.shap_values(instance_to_explain, nsamples=nsamples, silent=True)
    
    # KernelExplainer returns list for multi-class, or array depending on output shape
    if isinstance(shap_vals, list):
        # Extract the positive class attributions
        # For binary probability, shap_vals is of length 2 or 1.
        shap_values = shap_vals[0]
    else:
        shap_values = shap_vals
        
    # Squeeze dimensions to get a flat 1D array corresponding to sequence positions
    shap_values = np.squeeze(shap_values)
    base_value = float(explainer.expected_value)
    
    return shap_values, base_value
