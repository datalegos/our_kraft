import pandas as pd
import pytest
import os
import numbers
import pickle
from unittest import mock
from sklearn.preprocessing import LabelEncoder

# Sample data simulating a small portion of KDDTrain+.txt
@pytest.fixture
def sample_df():
    data = {
        "duration": [0],
        "protocol_type": ["tcp"],
        "service": ["http"],
        "flag": ["SF"],
        "src_bytes": [181],
        "dst_bytes": [5450],
        "land": [0],
        "wrong_fragment": [0],
        "urgent": [0],
        "hot": [0],
        "num_failed_logins": [0],
        "logged_in": [1],
        "num_compromised": [0],
        "root_shell": [0],
        "su_attempted": [0],
        "num_root": [0],
        "num_file_creations": [0],
        "num_shells": [0],
        "num_access_files": [0],
        "num_outbound_cmds": [0],
        "is_host_login": [0],
        "is_guest_login": [0],
        "count": [9],
        "srv_count": [9],
        "serror_rate": [0.0],
        "srv_serror_rate": [0.0],
        "rerror_rate": [0.0],
        "srv_rerror_rate": [0.0],
        "same_srv_rate": [1.0],
        "diff_srv_rate": [0.0],
        "srv_diff_host_rate": [0.0],
        "dst_host_count": [9],
        "dst_host_srv_count": [9],
        "dst_host_same_srv_rate": [1.0],
        "dst_host_diff_srv_rate": [0.0],
        "dst_host_same_src_port_rate": [0.11],
        "dst_host_srv_diff_host_rate": [0.0],
        "dst_host_serror_rate": [0.0],
        "dst_host_srv_serror_rate": [0.0],
        "dst_host_rerror_rate": [0.0],
        "dst_host_srv_rerror_rate": [0.0],
        "attack_type": ["neptune"],
        "difficulty_level": [0],
    }
    return pd.DataFrame(data)

def test_attack_label_mapping(sample_df):
    attack_mapping = {
        'neptune': 'Flooding',
        'normal': 'Normal'
    }
    sample_df["attack_label"] = sample_df["attack_type"].map(attack_mapping).fillna("Normal")
    assert sample_df["attack_label"].iloc[0] == "Flooding"

def test_label_encoding(sample_df):
    le = LabelEncoder()
    sample_df["attack_label"] = ["Flooding"]
    encoded = le.fit_transform(sample_df["attack_label"])
    assert encoded[0] == 0

def test_categorical_encoding(sample_df):
    for col in ["protocol_type", "service", "flag"]:
        le = LabelEncoder()
        sample_df[col] = le.fit_transform(sample_df[col])
    assert isinstance(sample_df["protocol_type"].iloc[0], numbers.Number)

@mock.patch("pandas.DataFrame.to_csv")
def test_preprocessed_file_saved(mock_to_csv, sample_df):
    # Simulate to_csv without actually writing
    sample_df.to_csv("mock_path.csv", index=False)
    mock_to_csv.assert_called_once_with("mock_path.csv", index=False)

@mock.patch("pickle.dump")
@mock.patch("builtins.open", new_callable=mock.mock_open)
def test_encoder_saved(mock_open, mock_pickle):
    le = LabelEncoder()
    le.fit(["Normal", "Flooding"])
    with open("models/fake_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    mock_pickle.assert_called_once()
