import sys
import os
import pytest
import pandas as pd
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.visualizer import Visualizer

@pytest.fixture
def visualizer():
    return Visualizer()

def test_plot_sentiment_distribution(visualizer, tmp_path):
    data = pd.DataFrame({'sentiment': ['Positive', 'Negative', 'Neutral', 'Positive']})
    visualizer.output_dir = tmp_path
    visualizer.plot_sentiment_distribution(data)
    assert (tmp_path / "sentiment_distribution.png").exists()

def test_plot_toxicity_counts(visualizer, tmp_path):
    data = pd.DataFrame({'is_extremist': [0, 1, 0, 1]})
    visualizer.output_dir = tmp_path
    visualizer.plot_toxicity_counts(data)
    assert (tmp_path / "toxicity_counts.png").exists()

def test_plot_user_clusters(visualizer, tmp_path):
    data = pd.DataFrame({
        'user_id': [1, 2, 3],
        'avg_toxicity': [0.1, 0.9, 0.5],
        'review_count': [10, 5, 8],
        'negative_ratio': [0.2, 0.8, 0.4],
        'cluster': [0, 1, 0]
    })
    visualizer.output_dir = tmp_path
    visualizer.plot_user_clusters(data)
    assert (tmp_path / "user_clusters.png").exists()

def test_generate_wordcloud(visualizer, tmp_path):
    data = pd.DataFrame({
        'is_extremist': [1, 0, 1],
        'cleaned_review': ["hate bad awful", "nice good", "terrible hate"]
    })
    visualizer.output_dir = tmp_path
    visualizer.generate_wordcloud(data)
    assert (tmp_path / "toxic_wordcloud.png").exists()