# logger.py

import logging
import os

def setup_logger(log_file="training.log"):
    logger = logging.getLogger("MedicinalHerbsTrainer")
    logger.setLevel(logging.INFO)

    # Avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # File Handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # Add handlers
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger
