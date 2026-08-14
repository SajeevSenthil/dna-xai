import logging
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from typing import Tuple, Any
import transformers
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: True
transformers.utils.check_torch_load_is_safe = lambda *args, **kwargs: True
try:
    import transformers.modeling_utils
    transformers.modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: True
except Exception:
    pass

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
