import os
import pickle
import pandas as pd
import pytest

# Resolve paths relative to this test file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'random_forest_model.pkl')
ENCODER_PATH = os.path.join(PROJECT_ROOT, 'models', 'attack_label_encoder.pkl')
FEATURES_PATH = os.path.join(PROJECT_ROOT, 'models', 'feature_columns.pkl')

# Debug prints (optional - can be removed if not needed)
print("Random Forest Model Exists:", os.path.exists(MODEL_PATH))
print("Label Encoder Exists:", os.path.exists(ENCODER_PATH))
print("Feature Columns Exists:", os.path.exists(FEATURES_PATH))

# Fixtures to load model components
@pytest.fixture(scope="module")
def model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@pytest.fixture(scope="module")
def label_encoder():
    with open(ENCODER_PATH, 'rb') as f:
        return pickle.load(f)

@pytest.fixture(scope="module")
def feature_columns():
    with open(FEATURES_PATH, 'rb') as f:
        return pickle.load(f)

# Fixture for base input
@pytest.fixture
def base_input():
    return {
        'duration': 0,
        'src_bytes': 491,
        'dst_bytes': 0,
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': 0,
        'num_failed_logins': 0,
        'protocol_type_tcp': 1,
        'protocol_type_udp': 0,
        'service_http': 0,
        'service_ftp': 0,
        'flag_SF': 1,
        'flag_REJ': 0,
    }

# Helper function
def prepare_input(sample_input, feature_columns):
    df = pd.DataFrame([sample_input])
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        zeros = pd.DataFrame(0, index=df.index, columns=missing_cols)
        df = pd.concat([df, zeros], axis=1)
    df = df[feature_columns]
    return df

# 1. Test if all components load correctly
def test_load_model_and_metadata(model, label_encoder, feature_columns):
    assert model is not None
    assert label_encoder is not None
    assert isinstance(feature_columns, list)
    assert len(feature_columns) > 0

# 2. Test if base input can be used to form a valid DataFrame
def test_prepare_input_columns(base_input, feature_columns):
    df = prepare_input(base_input, feature_columns)
    assert list(df.columns) == feature_columns
    assert df.isnull().sum().sum() == 0  # No missing values

# 3. Test prediction returns a valid label
def test_prediction_label_valid(model, label_encoder, feature_columns, base_input):
    df = prepare_input(base_input, feature_columns)
    pred_encoded = model.predict(df)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    assert isinstance(pred_label, str)
    assert pred_label in label_encoder.classes_

# 4. Test model prediction on variant inputs
@pytest.mark.parametrize("variant_input", [
    {'duration': 5, 'src_bytes': 100, 'dst_bytes': 200, 'protocol_type_udp': 1, 'protocol_type_tcp': 0, 'service_ftp': 1, 'flag_REJ': 1},
    {'duration': 0, 'src_bytes': 0, 'dst_bytes': 0, 'protocol_type_tcp': 1, 'protocol_type_udp': 0, 'service_http': 1, 'flag_SF': 1},
    {'duration': 2, 'src_bytes': 300, 'dst_bytes': 150, 'hot': 2, 'urgent': 1, 'protocol_type_tcp': 1, 'flag_SF': 1},
])
def test_variant_predictions(model, label_encoder, feature_columns, variant_input):
    df = prepare_input(variant_input, feature_columns)
    pred_encoded = model.predict(df)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    assert pred_label in label_encoder.classes_

# 5. Test that missing input values are correctly filled
def test_missing_values_handled(feature_columns):
    incomplete_input = {'duration': 1, 'src_bytes': 10}
    df = prepare_input(incomplete_input, feature_columns)
    assert list(df.columns) == feature_columns
    assert df.isnull().sum().sum() == 0
