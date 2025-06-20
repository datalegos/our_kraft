import os
import pickle

# Get absolute paths for the model files relative to this test file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'random_forest_model.pkl')
ENCODER_PATH = os.path.join(PROJECT_ROOT, 'models', 'label_encoder.pkl')
FEATURES_PATH = os.path.join(PROJECT_ROOT, 'models', 'feature_columns.pkl')


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"


def test_encoder_file_exists():
    assert os.path.exists(ENCODER_PATH), f"Label encoder file not found at {ENCODER_PATH}"


def test_feature_columns_file_exists():
    assert os.path.exists(FEATURES_PATH), f"Feature columns file not found at {FEATURES_PATH}"


def test_model_can_be_loaded():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    assert model is not None, "Failed to load the model"


def test_encoder_can_be_loaded():
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
    assert encoder is not None, "Failed to load the label encoder"


def test_feature_columns_format():
    with open(FEATURES_PATH, 'rb') as f:
        feature_columns = pickle.load(f)
    assert isinstance(feature_columns, list), "Feature columns should be a list"
    assert len(feature_columns) > 0, "Feature columns list is empty"


if __name__ == "__main__":
    # For quick local run without pytest
    print("MODEL_PATH:", MODEL_PATH)
    print("ENCODER_PATH:", ENCODER_PATH)
    print("FEATURES_PATH:", FEATURES_PATH)
