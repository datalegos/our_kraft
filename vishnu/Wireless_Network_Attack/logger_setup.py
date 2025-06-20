# logger_setup.py
import logging
import os

def setup_logging(log_file_path=None):
    """
    Setup logging configuration.
    If log_file_path is provided, logs will be saved to that file.
    Logs also appear in the console (terminal).
    """
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    if log_file_path:
        logging.basicConfig(
            level=logging.DEBUG,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file_path, mode='w'),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.DEBUG,
            format=log_format,
            handlers=[logging.StreamHandler()]
        )
