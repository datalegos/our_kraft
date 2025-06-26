# tests/test_config_loader.py

import pytest
from medicinal_herbs_app.config_loader import load_config

def test_load_config():
    # Attempt to load config — fail the test if exception is raised
    try:
        config = load_config()
    except Exception as e:
        pytest.fail(f"Loading config failed with exception: {e}")

    # Check for required top-level keys
    required_keys = ["data", "model", "groq", "logging"]
    for key in required_keys:
        assert key in config, f"Missing required key in config: '{key}'"

    # Optional: deeper checks (example — check that required subkeys exist)
    assert "data_dir" in config["data"], "Missing 'data_dir' in 'data' section"
    assert "model_save_path" in config["model"], "Missing 'model_save_path' in 'model' section"
    assert "api_key" in config["groq"], "Missing 'api_key' in 'groq' section"
    assert "model_name" in config["groq"], "Missing 'model_name' in 'groq' section"
    assert "log_file" in config["logging"], "Missing 'log_file' in 'logging' section"
    assert "log_level" in config["logging"], "Missing 'log_level' in 'logging' section"