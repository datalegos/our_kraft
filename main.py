# main.py
# Purpose: Orchestrate the review analysis pipeline
# Author: Naveen
# Date: June 12, 2025

import sys
import os
from pathlib import Path
import logging

# Add project root to sys.path for module resolution
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from scripts.cleaner import TextCleaner
from scripts.classifier import ReviewClassifier
from scripts.clustering import UserClustering
from scripts.visualizer import Visualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
RAW_DATA_PATH = Path("data/raw/reviews.csv")
PROCESSED_DIR = Path("data/processed")
CLEANED_FILE = PROCESSED_DIR / "cleaned_reviews.csv"
CLASSIFIED_FILE = PROCESSED_DIR / "classified_reviews.csv"
CLUSTERED_FILE = PROCESSED_DIR / "clustered_users.csv"

def run_pipeline():
    """Run the full extremist reviewer detection pipeline."""
    try:
        logger.info("🔍 Starting review analysis pipeline...")

        # Ensure output folder exists
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        # Initialize pipeline components
        cleaner = TextCleaner()
        classifier = ReviewClassifier()
        clustering = UserClustering()
        visualizer = Visualizer()

        # Step 1: Load and clean data
        logger.info("🧹 Cleaning reviews from %s", RAW_DATA_PATH)
        # Specify correct separator for your CSV (comma)
        cleaned_df = cleaner.preprocess_dataset(RAW_DATA_PATH, CLEANED_FILE, sep=',')
        if cleaned_df is None or cleaned_df.empty:
            raise ValueError("❌ Cleaning failed. Check input format and content.")

        # Optional: Train the model if labeled data is available
        if 'label' in cleaned_df.columns:
            logger.info("🎓 Training classifier...")
            classifier.train_classifier(cleaned_df, text_column='cleaned_review', label_column='label')
        else:
            logger.warning("⚠️ No 'label' column found. Skipping training and using rule-based classification.")

        # Step 2: Classify reviews using trained or rule-based model
        logger.info("🔍 Classifying reviews...")
        classified_df = classifier.classify_reviews(CLEANED_FILE, CLASSIFIED_FILE)
        if classified_df is None or classified_df.empty:
            raise ValueError("❌ Classification failed.")

        # Step 3: Cluster users based on extremist behavior
        logger.info("🧠 Clustering users...")
        clustered_df = clustering.cluster_users(classified_df, CLUSTERED_FILE) # type: ignore
        if clustered_df is None or clustered_df.empty:
            raise ValueError("❌ Clustering failed.")

        # Step 4: Visualize results
        logger.info("📊 Generating visualizations...")
        visualizer.plot_sentiment_distribution(classified_df)
        visualizer.plot_toxicity_counts(classified_df)
        visualizer.plot_user_clusters(clustered_df)
        visualizer.generate_wordcloud(classified_df)

        logger.info("✅ Pipeline completed successfully.")

    except Exception as e:
        logger.error("🚨 Pipeline failed: %s", str(e))
        raise

if __name__ == "__main__":
    run_pipeline()