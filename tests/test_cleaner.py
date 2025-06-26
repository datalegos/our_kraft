import sys
import os
import pytest
import pandas as pd
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.cleaner import TextCleaner

@pytest.fixture
def cleaner():
    return TextCleaner()

def test_clean_text(cleaner):
    assert cleaner.clean_text("This is <b>great</b>! 😊") == "great"
    assert cleaner.clean_text("I can't believe it!") == "ca believe"
    assert cleaner.clean_text("Running, runner, ran.") == "running runner ran"

def test_preprocess_dataset(cleaner, tmp_path):
    data = pd.DataFrame({
        'user_id': [1, 2],
        'review': ["Great product!", "Terrible, I HATE it <b>so much</b>"],
        'timestamp': ['2023-01-01', '2023-01-02']
    })
    input_path = tmp_path / "test_reviews.csv"
    output_path = tmp_path / "cleaned_test_reviews.csv"
    data.to_csv(input_path, index=False)
    result_df = cleaner.preprocess_dataset(input_path, output_path)
    assert 'cleaned_review' in result_df.columns
    assert result_df['cleaned_review'].iloc[0] == "great product"
    assert (cleaner.processed_dir / output_path.name).exists()