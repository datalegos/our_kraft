import sys
import os
import pytest
import pandas as pd
import logging
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.classifier import ReviewClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@pytest.fixture
def classifier():
    return ReviewClassifier()

def test_analyze_sentiment(classifier):
    test_texts = [
        ("Great product, love it!", "Positive"),
        ("This is terrible, hate it!", "Negative"),
        ("It's okay, nothing special.", "Neutral"),
    ]
    for text, expected in test_texts:
        result = classifier.analyze_sentiment(text)
        assert result == expected, f"Expected '{expected}' for '{text}', but got '{result}'"

def test_classify_reviews(classifier, tmp_path):
    data = pd.DataFrame({
        'user_id': [1, 2, 3],
        'cleaned_review': ["great product", "hate terrible product", "okay product"]
    })
    input_path = tmp_path / "test_reviews.csv"
    output_path = tmp_path / "classified_test_reviews.csv"
    data.to_csv(input_path, index=False)
    result_df = classifier.classify_reviews(input_path, output_path)
    output_df = pd.read_csv(output_path)
    assert 'sentiment' in output_df.columns
    assert 'is_extremist' in output_df.columns
    assert output_df.loc[0, 'sentiment'] == "Positive"
    assert output_df.loc[1, 'sentiment'] == "Negative" or output_df.loc[1, 'is_extremist'] == 1
    assert output_df.loc[2, 'sentiment'] == "Neutral"
    assert output_df.loc[2, 'is_extremist'] == 0

def test_analyze_sentiment_edge_cases(classifier):
    test_texts = [
        ("", "Neutral"),
        ("I have no opinion.", "Neutral"),
        ("Absolutely fantastic!", "Positive"),
        ("Absolutely horrible!", "Negative"),
        ("meh", "Neutral"),
        ("dislike", "Negative"),
        ("I like it", "Positive"),
        ("Could be better", "Neutral"),
        ("Worst ever", "Negative"),
        ("Best ever", "Positive"),
    ]
    for text, expected in test_texts:
        result = classifier.analyze_sentiment(text)
        assert result == expected, f"Expected '{expected}' for '{text}', but got '{result}'"

def test_classify_reviews_missing_column(classifier, tmp_path):
    data = pd.DataFrame({'user_id': [1, 2, 3], 'review': ["good", "bad", "okay"]})
    input_path = tmp_path / "missing_col.csv"
    output_path = tmp_path / "output.csv"
    data.to_csv(input_path, index=False)
    with pytest.raises(ValueError, match="Input file must contain a 'cleaned_review' column."):
        classifier.classify_reviews(input_path, output_path)

def test_classify_reviews_rule_based_extremist(classifier, tmp_path):
    data = pd.DataFrame({
        'user_id': [1, 2, 3],
        'cleaned_review': ["I hate this", "This is awful", "Nice and lovely"]
    })
    input_path = tmp_path / "rule_based.csv"
    output_path = tmp_path / "classified_rule_based.csv"
    data.to_csv(input_path, index=False)
    output_df = classifier.classify_reviews(input_path, output_path)
    assert output_df.loc[0, 'is_extremist'] == 1
    assert output_df.loc[1, 'is_extremist'] == 1
    assert output_df.loc[2, 'is_extremist'] == 0