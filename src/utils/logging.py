import logging
import os
import sys
from typing import Optional

def setup_logging(
    log_file: Optional[str] = None, 
    level: int = logging.INFO
) -> None:
    """
    Sets up global logging configuration.
    Outputs to both stdout and a log file if provided.
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
        
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )
    
    # Suppress verbose logger outputs from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    
    logging.info("Logging configured successfully.")
