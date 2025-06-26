# app.py
# Purpose: Streamlit dashboard for visualizing review analysis results
# Author: Expert Developer
# Date: June 10, 2025

import streamlit as st
import pandas as pd
from PIL import Image
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load processed data from CSV files."""
    try:
        data_dir = Path('data/processed')
        reviews_df = pd.read_csv(data_dir / 'classified_reviews.csv')
        clusters_df = pd.read_csv(data_dir / 'clustered_users.csv')
        logger.info("Successfully loaded data.")
        return reviews_df, clusters_df
    except FileNotFoundError as e:
        logger.error(f"Data files not found: {str(e)}")
        st.error("Data files not found. Please run the pipeline first.")
        return None, None
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return None, None

from typing import Optional

def load_plot(plot_name: str) -> Optional[Image.Image]:
    """Load a plot image from the dashboard/plots directory."""
    try:
        plot_path = Path('dashboard/plots') / plot_name
        img = Image.open(plot_path)
        logger.info(f"Loaded plot: {plot_name}")
        return img
    except FileNotFoundError:
        logger.error(f"Plot not found: {plot_name}")
        st.error(f"Plot not found: {plot_name}")
        return None
    except Exception as e:
        logger.error(f"Error loading plot {plot_name}: {str(e)}")
        st.error(f"Error loading plot {plot_name}: {str(e)}")
        return None

def main():
    """Main function to render the Streamlit dashboard."""
    st.set_page_config(page_title="Extremist Reviewer Detection", layout="wide")
    st.title("Extremist Reviewer Detection Dashboard")
    st.markdown("Visualize sentiment, toxicity, and user clustering of reviews.")

    # Load data
    reviews_df, clusters_df = load_data()
    if reviews_df is None or clusters_df is None:
        return

    # Debug columns
    st.sidebar.markdown("**Debug Info**")
    st.sidebar.write("Reviews Columns:", list(reviews_df.columns))
    st.sidebar.write("Clusters Columns:", list(clusters_df.columns))

    # Sidebar for navigation
    section = st.sidebar.selectbox(
        "Select Section",
        ["Overview", "Sentiment Analysis", "Toxicity Analysis", "User Clustering", "Toxic Word Cloud"]
    )

    # Overview Section
    if section == "Overview":
        st.header("Overview")
        st.write("Total Reviews:", len(reviews_df))
        st.write("Unique Users:", reviews_df['user_id'].nunique())

        if 'is_extremist' in reviews_df.columns:
            st.write("Toxic Reviews:", len(reviews_df[reviews_df['is_extremist'] == 1]))
        else:
            st.warning("Column 'is_extremist' not found in reviews_df!")

        if 'cluster' in clusters_df.columns:
            st.write("User Clusters:", clusters_df['cluster'].nunique())
        else:
            st.warning("Column 'cluster' not found in clusters_df!")

    # Sentiment Analysis
    elif section == "Sentiment Analysis":
        st.header("Sentiment Analysis")
        img = load_plot('sentiment_distribution.png')
        if img:
            st.image(img, caption="Sentiment Distribution", use_column_width=True)
        if 'sentiment' in reviews_df.columns:
            st.subheader("Sentiment Breakdown")
            sentiment_counts = reviews_df['sentiment'].value_counts()
            st.dataframe(sentiment_counts)
        else:
            st.warning("Column 'sentiment' not found in reviews_df!")

    # Toxicity Analysis
    elif section == "Toxicity Analysis":
        st.header("Toxicity Analysis")
        img = load_plot('toxicity_counts.png')
        if img:
            st.image(img, caption="Toxic vs Non-Toxic Reviews", use_column_width=True)
        if 'is_extremist' in reviews_df.columns:
            st.subheader("Toxicity Details")
            toxicity_counts = reviews_df['is_extremist'].value_counts().rename({0: 'Non-Toxic', 1: 'Toxic'})
            st.dataframe(toxicity_counts)
        else:
            st.warning("Column 'is_extremist' not found in reviews_df!")

    # User Clustering
    elif section == "User Clustering":
        st.header("User Clustering")
        img = load_plot('user_clusters.png')
        if img:
            st.image(img, caption="User Clusters", use_column_width=True)
        if clusters_df is not None:
            st.subheader("Cluster Details")
            st.dataframe(clusters_df)

    # Toxic Word Cloud
    elif section == "Toxic Word Cloud":
        st.header("Toxic Word Cloud")
        img = load_plot('toxic_wordcloud.png')
        if img:
            st.image(img, caption="Word Cloud of Toxic Reviews", use_column_width=True)

if __name__ == "__main__":
    main()
