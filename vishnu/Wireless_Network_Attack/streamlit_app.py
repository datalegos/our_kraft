# streamlit_app.py
import os
import logging
import pandas as pd
import streamlit as st
from predict import predict_attack
from logger_setup import setup_logging


# Ensure output folder exists
os.makedirs("outputs", exist_ok=True)

# Setup application logger
app_logger = logging.getLogger("app_logger")
app_logger.setLevel(logging.INFO)
app_log_file = os.path.join("outputs", "app.log")

# Prevent adding duplicate handlers
if not app_logger.hasHandlers():
    app_handler = logging.FileHandler(app_log_file)
    app_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    app_logger.addHandler(app_handler)
    app_logger.addHandler(logging.StreamHandler())

app_logger.info("🔧 Streamlit app initialized.")

# Streamlit UI
st.set_page_config(page_title="Wireless Network Attack Prediction System", layout="wide")
st.title("🚨 Wireless Network Attack Prediction System")
st.markdown("This app uses a trained ML model to detect potential attacks based on network traffic features.")

# Dropdown options for categorical fields
PROTOCOLS = ['tcp', 'udp', 'icmp']
SERVICES = ['http', 'domain_u', 'smtp', 'ftp', 'eco_i', 'private', 'other']
FLAGS = ['SF', 'S0', 'REJ', 'RSTR', 'SH', 'S1', 'S2', 'S3']

# Sample input data for demo/test
sample_inputs = {
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
        'dst_host_srv_rerror_rate': 0.00
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
        'dst_host_srv_rerror_rate': 0.0
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
        'duration': 0, 'protocol_type': 'icmp', 'service': 'eco_i', 'flag': 'SF',
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
        'dst_host_rerror_rate': 0.00, 'dst_host_srv_rerror_rate': 0.00
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
        'dst_host_srv_rerror_rate': 0.0
    }
}

# Input choice
preset_choice = st.selectbox("🧪 Choose a test case (or select Manual Input)", ["Manual Input"] + list(sample_inputs.keys()))

# Input handling
if preset_choice != "Manual Input":
    input_data = sample_inputs[preset_choice]
    st.success(f"✅ Auto-filled data for {preset_choice}")
else:
    input_data = {}
    st.markdown("### 🛠️ Enter Network Features Manually")
    input_data['duration'] = st.number_input("Duration", value=0)
    input_data['protocol_type'] = st.selectbox("Protocol Type", PROTOCOLS)
    input_data['service'] = st.selectbox("Service", SERVICES)
    input_data['flag'] = st.selectbox("Flag", FLAGS)

    numeric_features = [
        'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
        'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count',
        'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
    ]
    for feature in numeric_features:
        if "rate" in feature:
            input_data[feature] = st.number_input(feature, min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        else:
            input_data[feature] = st.number_input(feature, min_value=0, value=0)

# Prediction
if st.button("🔍 Predict Attack Type"):
    st.write("🟡 Running prediction...")
    try:
        prediction, info = predict_attack(input_data)

        # Show result
        st.success(f"✅ Prediction result: {prediction}")
        color = "green" if prediction == "Normal" else "red"
        st.markdown(f"<h3 style='color:{color}'>🛡️ Predicted Attack Type: {prediction}</h3>", unsafe_allow_html=True)

        if info:
            st.markdown(f"**🛡️ Prevention Tip:** {info.get('prevention', 'N/A')}")
            st.markdown(f"**🧯 Precaution Tip:** {info.get('precautions', 'N/A')}")
        else:
            st.info("ℹ️ No additional prevention or precaution info available.")

        # Save to prediction log
        with open("outputs/prediction_log.txt", "a") as log_file:
            log_file.write(f"Prediction Input: {input_data}\n")
            log_file.write(f"Predicted Attack Type: {prediction}\n")
            if info:
                log_file.write(f"Prevention: {info.get('prevention', '')}\n")
                log_file.write(f"Precaution: {info.get('precautions', '')}\n")
            log_file.write("-" * 60 + "\n")

        app_logger.info(f"Prediction completed: {prediction}")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        app_logger.exception("Prediction failed due to exception.")
