import os
import yaml
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

def load_config(config_path="medicinal_herbs_app/config.yaml"):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Convert paths to absolute where needed
        project_root = os.path.dirname(os.path.abspath(config_path))

        # Update classes_file to absolute path
        classes_file = config["model"].get("classes_file")
        if classes_file:
            config["model"]["classes_file"] = os.path.abspath(os.path.join(project_root, "..", classes_file))

        # Similarly, you can add this for model_save_path if you want
        model_save_path = config["model"].get("model_save_path")
        if model_save_path:
            config["model"]["model_save_path"] = os.path.abspath(os.path.join(project_root, "..", model_save_path))

        logger.info(f"Configuration loaded successfully from {os.path.abspath(config_path)}.")
        return config

    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise
