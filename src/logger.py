import logging
import os
import sys
from datetime import datetime

# Function to get absolute paths within the project
import os


def get_project_path(directory_name):
    """Get absolute path to a directory within the project"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    directory_path = os.path.join(project_root, directory_name)
    os.makedirs(directory_path, exist_ok=True)
    return directory_path


def setup_logger(name, log_dir=None):
    """Set up logger that writes to both console and file"""
    # Use absolute path within the project directory
    if log_dir is None:
        log_dir = get_project_path('logs')
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create a unique log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file
