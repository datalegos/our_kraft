# scripts/classifier.py
# Purpose: Classify reviews as toxic or non-toxic
# Author: Naveen
# Date: June 10, 2025

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
import logging
from pathlib import Path
from typing import Union, Dict, Any
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data with error handling
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReviewClassifier:
    """Class to handle sentiment and toxicity classification."""

    def __init__(self):
        self.data_dir = Path('data/processed')
        self.model_dir = Path('models')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better analysis."""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        try:
            # Tokenize and lemmatize
            tokens = word_tokenize(text.lower())
            lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
            return " ".join(lemmatized)
        except Exception as e:
            logger.warning(f"Error preprocessing text: {e}")
            return text.lower()

    def analyze_sentiment(self, text: str) -> str:
        """Improved sentiment analysis using lemmatization and substring matching."""
        if not isinstance(text, str) or not text.strip():
            return "Neutral"

        positive_keywords = {
            "great", "love", "excellent", "good", "amazing", "wonderful", "best", "awesome", 
            "like", "fantastic", "superb", "nice", "positive", "brilliant", "outstanding"
        }
        negative_keywords = {
            "terrible", "hate", "bad", "awful", "worst", "poor", "disappointing", "horrible", 
            "dislike", "boring", "disgusting", "pathetic", "useless"
        }
        neutral_keywords = {
            "okay", "meh", "fine", "average", "normal", "alright", "decent"
        }

        text_lower = text.lower().strip()
        
        # Check for explicit neutral phrases first
        for phrase in neutral_keywords:
            if phrase in text_lower:
                return "Neutral"

        # Check for positive and negative phrases
        positive_score = sum(1 for phrase in positive_keywords if phrase in text_lower)
        negative_score = sum(1 for phrase in negative_keywords if phrase in text_lower)

        # If no direct matches, try lemmatized tokens
        if positive_score == 0 and negative_score == 0:
            try:
                tokens = word_tokenize(text_lower)
                lemmas = {self.lemmatizer.lemmatize(word) for word in tokens if word.isalpha()}
                
                positive_score = len(lemmas & positive_keywords)
                negative_score = len(lemmas & negative_keywords)
            except Exception as e:
                logger.warning(f"Error in lemmatization: {e}")

        # Determine sentiment based on scores
        if positive_score > negative_score:
            return "Positive"
        elif negative_score > positive_score:
            return "Negative"
        else:
            return "Neutral"

    def train_classifier(
        self, 
        df: pd.DataFrame, 
        text_column: str, 
        label_column: str
    ) -> Dict[str, Any]:
        """Train toxicity classifier and return evaluation metrics."""
        try:
            logger.info("Training classifier...")
            
            # Validate input data
            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in DataFrame")
            if label_column not in df.columns:
                logger.warning("⚠️ No 'label' column found. Skipping training and using rule-based classification.")
                return {}  # Skip training and return empty metrics
            
            # Clean and preprocess data
            df_clean = df.dropna(subset=[text_column, label_column])
            if len(df_clean) == 0:
                raise ValueError("No valid data after removing NaN values")
            
            # Ensure text is string type
            df_clean[text_column] = df_clean[text_column].astype(str)
            
            X = self.vectorizer.fit_transform(df_clean[text_column])
            y = df_clean[label_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train classifier
            self.classifier.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.classifier.predict(X_test)

            # Calculate metrics
            metrics = {
                'report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'accuracy': (y_pred == y_test).mean()
            }

            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

            # Save models
            with open(self.model_dir / 'vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
            with open(self.model_dir / 'classifier.pkl', 'wb') as f:
                pickle.dump(self.classifier, f)

            logger.info("Models saved successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training classifier: {str(e)}")
            raise

    def load_models(self) -> bool:
        """Load pre-trained models."""
        try:
            with open(self.model_dir / 'vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(self.model_dir / 'classifier.pkl', 'rb') as f:
                self.classifier = pickle.load(f)
            logger.info("Models loaded successfully")
            return True
        except FileNotFoundError:
            logger.warning("No trained models found")
            return False
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def predict_toxicity(self, text: str) -> int:
        """Predict if a single text is toxic."""
        if not isinstance(text, str) or not text.strip():
            return 0
        
        try:
            X = self.vectorizer.transform([text])
            return self.classifier.predict(X)[0]
        except Exception as e:
            logger.warning(f"Error predicting toxicity: {e}")
            # Fallback to rule-based classification
            toxic_words = {'hate', 'terrible', 'awful', 'disgusting', 'pathetic', 'useless', 'stupid'}
            return 1 if any(word in text.lower().split() for word in toxic_words) else 0

    def classify_reviews(
        self, 
        input_path: Union[str, Path], 
        output_path: Union[str, Path]
    ) -> pd.DataFrame:
        """Classify reviews and save results."""
        try:
            input_path = Path(input_path)
            
            # Validate input file
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            logger.info(f"Reading dataset from {input_path}")
            df = pd.read_csv(input_path)
            
            if 'cleaned_review' not in df.columns:
                raise ValueError("Input file must contain a 'cleaned_review' column.")
            
            # Remove rows with NaN values in the review column
            df = df.dropna(subset=['cleaned_review'])
            df['cleaned_review'] = df['cleaned_review'].astype(str)
            
            logger.info(f"Processing {len(df)} reviews...")

            # Analyze sentiment
            logger.info("Analyzing sentiment...")
            df['sentiment'] = df['cleaned_review'].apply(self.analyze_sentiment)

            # Load models and predict toxicity
            model_loaded = self.load_models()
            
            if model_loaded:
                logger.info("Using trained model for toxicity classification...")
                try:
                    X = self.vectorizer.transform(df['cleaned_review'])
                    df['is_toxic'] = self.classifier.predict(X)
                except Exception as e:
                    logger.error(f"Error using trained model: {e}")
                    logger.info("Falling back to rule-based classification...")
                    df['is_toxic'] = df['cleaned_review'].apply(self.predict_toxicity)
            else:
                logger.info("Using rule-based toxicity classification...")
                toxic_words = {'hate', 'terrible', 'awful', 'disgusting', 'pathetic', 'useless', 'stupid'}
                df['is_toxic'] = df['cleaned_review'].apply(
                    lambda x: 1 if any(word in str(x).lower().split() for word in toxic_words) else 0
                )

            # Create output directory
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save results
            logger.info(f"Saving classified dataset to {output_path}")
            df.to_csv(output_path, index=False)
            
            # Log summary statistics
            sentiment_counts = df['sentiment'].value_counts()
            toxicity_counts = df['is_toxic'].value_counts()
            
            logger.info(f"Sentiment distribution: {sentiment_counts.to_dict()}")
            logger.info(f"Toxicity distribution: {toxicity_counts.to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error classifying reviews: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        classifier = ReviewClassifier()
        input_file = "data/processed/cleaned_reviews.csv"
        output_file = "data/processed/classified_reviews.csv"
        
        # Check if input file exists
        if not Path(input_file).exists():
            logger.error(f"Input file {input_file} does not exist")
        else:
            result_df = classifier.classify_reviews(input_file, output_file)
            logger.info("Classification completed successfully")
            
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        raise