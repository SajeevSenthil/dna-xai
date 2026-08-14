import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """
    Sets seed for standard python random, numpy, and PyTorch (CPU and CUDA)
    to ensure reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Configure PyTorch backends for strict determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set seed in transformers if imported
    try:
        from transformers import set_seed as tf_set_seed
        tf_set_seed(seed)
    except ImportError:
        pass

    logger.info(f"Random seed set to {seed} (reproducibility enabled).")
