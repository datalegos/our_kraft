import os
import pickle
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from logger_setup import setup_logging


# Configure logging to write logs to the outputs directory
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("outputs/app.log", mode='w'),
        logging.StreamHandler()
    ]
)

# Ensure output directory exists
# os.makedirs("outputs", exist_ok=True)
# predict_logger = setup_logger("predict", "outputs/prediction_log.txt")


# Paths to saved model and encoders
MODEL_PATH = 'models/random_forest_model.pkl'
ENCODER_PATH = 'models/attack_label_encoder.pkl'
CAT_ENCODERS_PATH = 'models/cat_encoders.pkl'
FEATURE_COLUMNS_PATH = 'models/feature_columns.pkl'

# Load model and encoders
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)
with open(CAT_ENCODERS_PATH, 'rb') as f:
    cat_encoders = pickle.load(f)
with open(FEATURE_COLUMNS_PATH, 'rb') as f:
    feature_columns = pickle.load(f)

logging.info("Model, encoders, and feature columns loaded successfully.")

# Attack prevention info
attack_info = {
    "Normal": {
        "prevention": "No attack detected.",
        "precautions": "Maintain standard security practices."
    },
    "Black Hole": {
        "prevention": "Use secure routing and watchdog timers.",
        "precautions": "Continuously monitor packet drops."
    },
    "Flooding": {
        "prevention": "Rate-limiting and firewall filtering.",
        "precautions": "Use intrusion detection systems."
    },
    "Grayhole": {
        "prevention": "Multipath routing and monitoring.",
        "precautions": "Detect and isolate misbehaving nodes."
    },
    "Scheduling": {
        "prevention": "Synchronize network schedules.",
        "precautions": "Use secure time-sync protocols."
    }
}

# Core prediction logic
def predict_attack(input_data):
    df = pd.DataFrame([input_data])

    # Drop unused columns
    for col in ['attack_type', 'difficulty_level']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=cat_encoders.keys())
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Predict
    pred = model.predict(df)[0]
    pred_label = label_encoder.inverse_transform([pred])[0]
    info = attack_info.get(pred_label, {"prevention": "N/A", "precaution": "N/A"})
    return pred_label, info



# Optional: local test run for verification
if __name__ == "__main__":
    test_inputs = {
        'Normal': {
        'duration': 0, 'protocol_type': 'tcp', 'service': 'smtp', 'flag': 'SF',
            'src_bytes': 616, 'dst_bytes': 330, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
            'hot': 0, 'num_failed_logins': 0, 'logged_in': 1, 'num_compromised': 0,
            'root_shell': 0, 'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0,
            'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0,
            'is_guest_login': 0, 'count': 1, 'srv_count': 2, 'serror_rate': 0.00,
            'srv_serror_rate': 0.00, 'rerror_rate': 0.00, 'srv_rerror_rate': 0.00,
            'same_srv_rate': 1.00, 'diff_srv_rate': 0.00, 'srv_diff_host_rate': 1.00,
            'dst_host_count': 255, 'dst_host_srv_count': 129, 'dst_host_same_srv_rate': 0.51,
            'dst_host_diff_srv_rate': 0.03, 'dst_host_same_src_port_rate': 0.00,
            'dst_host_srv_diff_host_rate': 0.00, 'dst_host_serror_rate': 0.00,
            'dst_host_srv_serror_rate': 0.00, 'dst_host_rerror_rate': 0.33,
            'dst_host_srv_rerror_rate': 0.00,"attack_type": 'normal', "difficulty_level": 18,
    },

    'Black Hole': {
        'duration': 0, 'protocol_type': 'tcp', 'service': 'ftp', 'flag': 'SF',
        'src_bytes': 235, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
        'hot': 1, 'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0,
        'root_shell': 0, 'su_attempted': 0, 'num_root': 0, 'num_file_creations': 1,
        'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0,
        'is_guest_login': 0, 'count': 36, 'srv_count': 4, 'serror_rate': 0.0,
        'srv_serror_rate': 0.0, 'rerror_rate': 0.0, 'srv_rerror_rate': 0.0,
        'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0,
        'dst_host_count': 127, 'dst_host_srv_count': 33, 'dst_host_same_srv_rate': 0.26,
        'dst_host_diff_srv_rate': 0.16, 'dst_host_same_src_port_rate': 0.14,
        'dst_host_srv_diff_host_rate': 0.27, 'dst_host_serror_rate': 0.0,
        'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.01,
        'dst_host_srv_rerror_rate': 0.0,
        'attack_type': 'warezclient', 'difficulty_level': 10
    },
    'Flooding': {
        'duration': 0, 'protocol_type': 'tcp', 'service': 'smtp', 'flag': 'REJ',
        'src_bytes': 0, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
        'hot': 0, 'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0,
        'root_shell': 0, 'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0,
        'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0,
        'is_host_login': 0, 'is_guest_login': 0, 'count': 511, 'srv_count': 511,
        'serror_rate': 1.0, 'srv_serror_rate': 1.0, 'rerror_rate': 0.0,
        'srv_rerror_rate': 0.0, 'same_srv_rate': 1.0, 'diff_srv_rate': 0.0,
        'srv_diff_host_rate': 0.0, 'dst_host_count': 255, 'dst_host_srv_count': 255,
        'dst_host_same_srv_rate': 1.0, 'dst_host_diff_srv_rate': 0.0,
        'dst_host_same_src_port_rate': 1.0, 'dst_host_srv_diff_host_rate': 0.0,
        'dst_host_serror_rate': 1.0, 'dst_host_srv_serror_rate': 1.0,
        'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0
    },
    'Grayhole': {
        'duration': 0, 'protocol_type': 'icpm', 'service': 'eco_i', 'flag': 'SF',
        'src_bytes': 8, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
        'hot': 0, 'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0,
        'root_shell': 0, 'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0,
        'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0,
        'is_host_login': 0, 'is_guest_login': 0, 'count': 1, 'srv_count': 18,
        'serror_rate': 0.0, 'srv_serror_rate': 0.0, 'rerror_rate': 0.0,
        'srv_rerror_rate': 0.0, 'same_srv_rate': 1.00, 'diff_srv_rate': 0.00,
        'srv_diff_host_rate': 1.00, 'dst_host_count': 1, 'dst_host_srv_count': 65,
        'dst_host_same_srv_rate': 1.00, 'dst_host_diff_srv_rate': 0.00,
        'dst_host_same_src_port_rate': 1.00, 'dst_host_srv_diff_host_rate': 0.51,
        'dst_host_serror_rate': 0.00, 'dst_host_srv_serror_rate': 0.00,
        'dst_host_rerror_rate': 0.00, 'dst_host_srv_rerror_rate': 0.00,'attack_type': 'ipsweep', 'difficulty_level': 18,
    },
    'Scheduling': {
      'duration': 0, 'protocol_type': 'tcp', 'service': 'smtp', 'flag': 'SF',
    'src_bytes': 269, 'dst_bytes': 403, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
    'hot': 0, 'num_failed_logins': 0, 'logged_in': 1, 'num_compromised': 0,
    'root_shell': 0, 'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0,
    'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0,
    'is_guest_login': 0, 'count': 10, 'srv_count': 15, 'serror_rate': 0.0,
    'srv_serror_rate': 0.0, 'rerror_rate': 0.0, 'srv_rerror_rate': 0.0,
    'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0,
    'dst_host_count': 255, 'dst_host_srv_count': 255, 'dst_host_same_srv_rate': 1.0,
    'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 1.0,
    'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0,
    'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0,
    'dst_host_srv_rerror_rate': 0.0,
    }
}

    for label, input_data in test_inputs.items():
        logging.info(f"\n{'='*30}\nTesting: {label}")
        prediction, info = predict_attack(input_data)
        logging.info(f"Predicted: {prediction} | Info: {info}")
