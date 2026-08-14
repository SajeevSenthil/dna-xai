import logging
from typing import Tuple, Any
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

def load_dnabert_base(model_name_or_path: str = "zhihan1996/DNABERT-2-117M") -> Tuple[Any, Any]:
    """
    Loads DNABERT-2 base encoder and BPE tokenizer.
    Enables trust_remote_code=True for custom BERT architectures.
    """
    logger.info(f"Loading DNABERT-2 from: {model_name_or_path}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True
        )
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {model_name_or_path}: {e}")
        raise e

    # Load model
    try:
        model = AutoModel.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True
        )
    except Exception as e:
        logger.error(f"Failed to load model from {model_name_or_path}: {e}")
        raise e

    logger.info("DNABERT-2 model and tokenizer loaded successfully.")
    return model, tokenizer
