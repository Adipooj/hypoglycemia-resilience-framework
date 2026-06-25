import logging
import sys
from pathlib import Path

def setup_logging(log_dir: Path = None, log_level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger that prints to standard output and optionally logs to a file.
    
    Args:
        log_dir (Path, optional): Directory to save log files. Defaults to None.
        log_level (int): Log level. Defaults to logging.INFO.
        
    Returns:
        logging.Logger: Configured root logger.
    """
    logger = logging.getLogger("hypo_resilience")
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
