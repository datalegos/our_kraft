# tests/test_client.py

import sys
import os

# Add the root directory of the project to sys.path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from medicinal_herbs_app.client import get_medicinal_info_groq
from medicinal_herbs_app.config_loader import load_config
import pytest
from medicinal_herbs_app.config_loader import load_config

config = load_config()

@pytest.mark.skipif(
    not (load_config().get("groq", {}).get("api_key") and load_config().get("groq", {}).get("api_key") != "YOUR_API_KEY_HERE"),
    reason="No valid Groq API key configured"
)
def test_get_medicinal_info_groq():
    herb_name = "Tulsi"

    response_text = get_medicinal_info_groq(herb_name)

    assert isinstance(response_text, str)
    assert len(response_text) > 0
    assert "Sorry, medicinal information" not in response_text