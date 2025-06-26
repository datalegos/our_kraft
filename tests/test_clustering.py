import sys
import os
import pytest
import pandas as pd
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.clustering import UserClustering

@pytest.fixture
def clustering():
    return UserClustering()

def test_cluster_users(clustering, tmp_path):
    data = pd.DataFrame({
        'user_id': [1, 2, 3, 1, 2, 3],
        'cleaned_review': ["great product", "hate terrible product", "okay product"] * 2,
        'sentiment': ["Positive", "Negative", "Neutral"] * 2,
        'is_extremist': [0, 1, 0, 0, 1, 0]
    })
    output_path = tmp_path / "clustered_test_users.csv"
    result_df = clustering.cluster_users(data, output_path)
    assert 'cluster' in result_df.columns
    assert output_path.exists()
    assert result_df.shape[0] == 3  # 3 unique users

def test_cluster_users_empty(clustering, tmp_path):
    data = pd.DataFrame(columns=['user_id', 'cleaned_review', 'sentiment', 'is_extremist'])
    output_path = tmp_path / "empty_clustered.csv"
    result_df = clustering.cluster_users(data, output_path)
    assert result_df.empty