import logging
import torch
import numpy as np
from sklearn.linear_model import Ridge
from typing import List, Any, Tuple

logger = logging.getLogger(__name__)

class DNALimeExplainer:
    """
    Custom LIME explainer tailored specifically for DNA sequences.
    1. Perturbs individual nucleotides by replacing them with 'N'.
    2. Measures predictions for each perturbed sequence.
    3. Fits a local weighted Ridge regression to estimate nucleotide importance.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: torch.device,
        max_length: int = 128
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def explain_sequence(
        self,
        sequence: str,
        num_perturbations: int = 150,
        perturbation_fraction: float = 0.15,
        kernel_width: float = 0.25
    ) -> Tuple[np.ndarray, float]:
        """
        Explains a single DNA sequence using local surrogate Ridge regression.
        
        Args:
            sequence: The DNA sequence string to explain.
            num_perturbations: Number of perturbed sequences to generate.
            perturbation_fraction: Probability of masking each position.
            kernel_width: Kernel width for Jaccard distance exponential weighting.
            
        Returns:
            lime_coefficients: Flat array containing the attribution score for each position.
            intercept: The base/intercept prediction of the local surrogate model.
        """
        seq_len = len(sequence)
        seq_list = list(sequence)
        
        # 1. Generate perturbations (binary masks)
        # Ensure the first perturbation is the original sequence (all 1s)
        masks = [np.ones(seq_len)]
        for _ in range(num_perturbations - 1):
            # Mask random positions with probability perturbation_fraction
            mask = np.random.binomial(1, 1 - perturbation_fraction, seq_len)
            # Ensure we don't accidentally mask the entire sequence
            if mask.sum() == 0:
                mask[np.random.randint(0, seq_len)] = 1
            masks.append(mask)
            
        masks = np.array(masks)
        
        # 2. Build perturbed DNA sequences and run predictions
        perturbed_sequences = []
        for mask in masks:
            perturbed_seq = "".join(
                [char if m == 1 else "N" for char, m in zip(seq_list, mask)]
            )
            perturbed_sequences.append(perturbed_seq)
            
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
                
        predictions = np.array(predictions)
        
        # 3. Calculate distance-based weights
        # Jaccard distance: fraction of elements that differ (which is fraction of masked elements)
        distances = 1.0 - (masks.sum(axis=1) / seq_len)
        # Exponential kernel weight
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
        
        # 4. Fit local weighted Ridge regression
        # X: binary masks (features)
        # y: model prediction probabilities (target)
        # w: similarity weights
        clf = Ridge(alpha=1.0, fit_intercept=True)
        clf.fit(masks, predictions, sample_weight=weights)
        
        # Coefficients represent local attribution weights
        lime_coeffs = clf.coef_
        intercept = float(clf.intercept_)
        
        return lime_coeffs, intercept

def compute_lime_explanations(
    sequence: str,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    max_length: int = 128,
    num_perturbations: int = 150
) -> Tuple[np.ndarray, float]:
    """
    Wrapper function to compute LIME explanations.
    """
    explainer = DNASimeExplainer(model, tokenizer, device, max_length)
    return explainer.explain_sequence(sequence, num_perturbations=num_perturbations)
