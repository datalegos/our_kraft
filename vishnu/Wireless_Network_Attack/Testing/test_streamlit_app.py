import os
import pickle
import pytest
import pandas as pd

# Set up paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'random_forest_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'attack_label_encoder.pkl')
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, 'models', 'feature_columns.pkl')

# Fixtures
@pytest.fixture(scope='module')
def model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@pytest.fixture(scope='module')
def label_encoder():
    with open(ENCODER_PATH, 'rb') as f:
        return pickle.load(f)

@pytest.fixture(scope='module')
def feature_columns():
    with open(FEATURE_COLUMNS_PATH, 'rb') as f:
        return pickle.load(f)

# Sample inputs mimicking Streamlit dropdowns
@pytest.mark.parametrize("user_input", [
    {
        'duration': 0,
        'protocol_type': 'udp',
        'service': 'other',
        'flag': 'SF',
        'src_bytes': 105,
        'dst_bytes': 146,
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': 0,
        'num_failed_logins': 0,
    },
    {
        'duration': 2,
        'protocol_type': 'tcp',
        'service': 'http',
        'flag': 'SF',
        'src_bytes': 300,
        'dst_bytes': 100,
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': 0,
        'num_failed_logins': 1,
    }
])
def test_streamlit_input_prediction(model, label_encoder, feature_columns, user_input):
    # One-hot encoding of categorical fields (protocol_type, service, flag)
    protocol_cols = [f'protocol_type_{val}' for val in ['tcp', 'udp', 'icmp']]
    service_cols = [f'service_{val}' for val in ['http', 'ftp', 'other']]
    flag_cols = [f'flag_{val}' for val in ['SF', 'REJ', 'S0']]

    row = {
        col: 0 for col in feature_columns  # initialize all to 0
    }

    # Fill in values
    for key, val in user_input.items():
        if key in ['protocol_type', 'service', 'flag']:
            encoded_key = f'{key}_{val}'
            if encoded_key in row:
                row[encoded_key] = 1
        else:
            row[key] = val

    df = pd.DataFrame([row])
    assert list(df.columns) == feature_columns

    prediction = model.predict(df)[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    assert predicted_label in label_encoder.classes_
    assert isinstance(predicted_label, str)
