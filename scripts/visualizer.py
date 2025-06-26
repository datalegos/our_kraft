# scripts/visualizer.py
# Purpose: Generate visualizations for review analysis
# Author: Naveen
# Date: June 10, 2025

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from wordcloud import WordCloud
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Visualizer:
    """Class to generate visualizations for review analysis."""
    
    def __init__(self):
        """Initialize visualizer with output directory."""
        self.data_dir = Path('data/processed')
        self.output_dir = Path('dashboard/plots')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme()

    def plot_sentiment_distribution(self, df: pd.DataFrame) -> None:
        """Plot pie chart of sentiment distribution."""
        try:
            logger.info("Generating sentiment distribution plot...")

            if 'sentiment' not in df.columns:
                logger.error("Missing 'sentiment' column in DataFrame.")
                return

            sentiment_counts = df['sentiment'].value_counts()
            plt.figure(figsize=(8, 6))
            plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140) # type: ignore
            plt.title('Sentiment Distribution of Reviews')
            plt.savefig(self.output_dir / 'sentiment_distribution.png', bbox_inches='tight')
            plt.close()
            logger.info("Sentiment distribution plot saved.")
        except Exception as e:
            logger.error(f"Error plotting sentiment distribution: {str(e)}")

    def plot_toxicity_counts(self, df: pd.DataFrame) -> None:
        """Plot bar chart of toxic vs non-toxic reviews."""
        try:
            logger.info("Generating toxicity counts plot...")

            if 'is_extremist' not in df.columns:
                logger.error("Missing 'is_extremist' column in DataFrame.")
                return

            toxicity_counts = df['is_extremist'].value_counts().reset_index()
            toxicity_counts.columns = ['Toxicity', 'Count']
            toxicity_counts['Toxicity'] = toxicity_counts['Toxicity'].map({0: 'Non-Toxic', 1: 'Toxic'})
            
            plt.figure(figsize=(8, 6))
            sns.barplot(data=toxicity_counts, x='Toxicity', y='Count')
            plt.title('Toxic vs Non-Toxic Reviews')
            plt.xlabel('Toxicity')
            plt.ylabel('Count')
            plt.savefig(self.output_dir / 'toxicity_counts.png', bbox_inches='tight')
            plt.close()
            logger.info("Toxicity counts plot saved.")
        except Exception as e:
            logger.error(f"Error plotting toxicity counts: {str(e)}")

    def plot_user_clusters(self, df: pd.DataFrame) -> None:
        """Plot scatter plot of user clusters."""
        try:
            logger.info("Generating user clusters plot...")

            required_columns = {'avg_toxicity', 'review_count', 'cluster', 'negative_ratio'}
            if not required_columns.issubset(df.columns):
                logger.error(f"Missing one or more required columns: {required_columns}")
                return

            plt.figure(figsize=(10, 8))
            sns.scatterplot(data=df, x='avg_toxicity', y='review_count',
                            hue='cluster', size='negative_ratio', palette='tab10', legend='brief')
            plt.title('User Clusters Based on Behavior')
            plt.xlabel('Average Toxicity Score')
            plt.ylabel('Review Count')
            plt.savefig(self.output_dir / 'user_clusters.png', bbox_inches='tight')
            plt.close()
            logger.info("User clusters plot saved.")
        except Exception as e:
            logger.error(f"Error in plotting user clusters: {str(e)}")

    def generate_wordcloud(self, df: pd.DataFrame) -> None:
        """Generate word cloud for toxic reviews."""
        try:
            logger.info("Generating word cloud for toxic reviews...")

            if 'is_extremist' not in df.columns or 'cleaned_review' not in df.columns:
                logger.error("Missing required columns for word cloud.")
                return

            toxic_reviews = ' '.join(df[df['is_extremist'] == 1]['cleaned_review'].dropna())
            if not toxic_reviews.strip():
                logger.warning("No toxic reviews found to generate word cloud.")
                return

            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(toxic_reviews)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud of Toxic Reviews')
            plt.savefig(self.output_dir / 'toxic_wordcloud.png', bbox_inches='tight')
            plt.close()
            logger.info("Word cloud plot saved.")
        except Exception as e:
            logger.error(f"Error generating word cloud: {str(e)}")


if __name__ == "__main__":
    visualizer = Visualizer()
    
    try:
        reviews_df = pd.read_csv("data/processed/classified_reviews.csv")
        clusters_df = pd.read_csv("data/processed/clustered_users.csv")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        exit(1)
    
    visualizer.plot_sentiment_distribution(reviews_df)
    visualizer.plot_toxicity_counts(reviews_df)
    visualizer.plot_user_clusters(clusters_df)
    visualizer.generate_wordcloud(reviews_df)