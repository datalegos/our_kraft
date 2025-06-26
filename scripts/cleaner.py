# scripts/cleaner.py
# Purpose: Preprocess and clean review text data for analysis
# Author: Naveen
# Date: June 10, 2025

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import emoji
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        logger.info("Downloading NLTK resources...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

download_nltk_data()

class TextCleaner:
    """Class to handle text preprocessing for review data."""
    
    def __init__(self):
        """Initialize cleaner with NLTK resources and directory paths."""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.data_dir = Path('data')
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def clean_text(self, text: str) -> str:
        """Clean a single review text by applying preprocessing steps."""
        try:
            text = str(text)
            text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
            text = emoji.demojize(text)  # Convert emojis to text
            contractions = {
                "can't": "ca not", "won't": "will not", "n't": " not",
                "'re": " are", "'ll": " will", "'ve": " have",
                "'d": " would", "'m": " am"
            }
            for contraction, expanded in contractions.items():
                text = text.replace(contraction, expanded)
            text = text.lower()  # Lowercase
            text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars
            tokens = word_tokenize(text)  # Tokenize
            tokens = [token for token in tokens if token not in self.stop_words]  # Remove stopwords
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize
            return ' '.join(tokens).strip()
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text

    def preprocess_dataset(self, input_path: str | Path, output_path: str | Path, sep: str = ',') -> pd.DataFrame:
        try:
            input_path_str = str(input_path)
            logger.info(f"Reading dataset from {input_path_str}")
            # Detect columns for your CSV
            df = pd.read_csv(input_path_str, sep=sep) if input_path_str.endswith('.csv') else pd.read_json(input_path_str)
            # Fix for your reviews.csv: columns are user_id,review,timestamp, but review column is not named 'review_text'
            # Rename 'review' to 'review_text' for compatibility
            if 'review' in df.columns:
                df = df.rename(columns={'review': 'review_text'})
            required_columns = ['user_id', 'review_text']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Dataset must contain columns: {required_columns}")
            df['cleaned_review'] = df['review_text'].apply(self.clean_text)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['user_id'] = df['user_id'].apply(lambda x: hash(str(x)) % 10**8)  # Anonymize
            df = df.dropna(subset=['cleaned_review']).drop_duplicates(subset=['user_id', 'cleaned_review'])
            # Always save to processed_dir
            output_file = self.processed_dir / Path(output_path).name
            output_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving cleaned dataset to {output_file}")
            df.to_csv(output_file, index=False)
            return df
        except Exception as e:
            logger.error(f"Error preprocessing dataset: {str(e)}")
            raise

if __name__ == "__main__":
    cleaner = TextCleaner()
    input_file = "data/raw/reviews.csv"
    output_file = "cleaned_reviews.csv"
    cleaner.preprocess_dataset(input_file, output_file)