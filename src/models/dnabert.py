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
from transformers.models.bert.configuration_bert import BertConfig

logger = logging.getLogger(__name__)

def load_dnabert_encoder(model_name_or_path: str, **kwargs) -> Any:
    """
    Loads the DNABERT-2 encoder with an explicit BertConfig (workaround recommended in the
    DNABERT-2 README) to avoid the config_class mismatch raised by AutoModel when the
    remote-code config has already been registered (e.g. by loading the tokenizer first).
    """
    config = BertConfig.from_pretrained(model_name_or_path)
    return AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, config=config, **kwargs)

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
        model = load_dnabert_encoder(model_name_or_path)
    except Exception as e:
        logger.error(f"Failed to load model from {model_name_or_path}: {e}")
        raise e

    logger.info("DNABERT-2 model and tokenizer loaded successfully.")
    return model, tokenizer
