import requests
import logging
from .config_loader import load_config

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

# Load configuration
config = load_config()

# Extract API config
api_key = config["groq"]["api_key"]
model_name = config["groq"]["model_name"]

# Example: Using GROQ API (you may need to adjust endpoint / payload)
def get_medicinal_info_groq(herb_name):
    try:
        prompt = f"What are the medicinal uses of the herb '{herb_name.replace('_', ' ')}'?"
        logger.info(f"Sending prompt to GROQ API: {prompt}")

        url = f"https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.7
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()
        reply = data["choices"][0]["message"]["content"].strip()
        logger.info(f"Received response from GROQ API")

        return reply

    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with GROQ API: {e}")
        return "Error fetching medicinal info."

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return "Unexpected error occurred."
