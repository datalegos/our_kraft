# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt

# # Load saved model, label encoder, and feature columns
# def load_artifacts():
#     with open('models/random_forest_model.pkl', 'rb') as f:
#         model = pickle.load(f)
#     with open('models/target_label_encoder.pkl', 'rb') as f:
#         label_encoder = pickle.load(f)
#     with open('models/feature_columns.pkl', 'rb') as f:
#         train_columns = pickle.load(f)
#     return model, label_encoder, train_columns

# # Preprocess input data to match training features
# def preprocess_input(user_input, train_columns):
#     df = pd.DataFrame([user_input])

#     # Categorical features to one-hot encode
#     cat_features = ['protocol_type', 'service', 'flag']

#     for col in cat_features:
#         if col in df.columns:
#             dummies = pd.get_dummies(df[col], prefix=col)
#             df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

#     # Add any missing columns from train_columns with 0
#     for col in train_columns:
#         if col not in df.columns:
#             df[col] = 0

#     # Drop extra columns not in train_columns
#     df = df[train_columns]

#     return df

# # Predict attack type
# def predict_attack(model, label_encoder, train_columns, user_input):
#     processed = preprocess_input(user_input, train_columns)
#     pred_encoded = model.predict(processed)[0]
#     pred_label = label_encoder.inverse_transform([pred_encoded])[0]
#     return pred_label

# # Sample inputs for auto-fill
# sample_inputs = {
#     'Black Hole': {
#         'duration': 0,
#         'protocol_type': 'tcp',
#         'service': 'http',
#         'flag': 'SF',
#         'src_bytes': 215,
#         'dst_bytes': 45076,
#         'land': 0,
#         'wrong_fragment': 0,
#         'urgent': 0,
#         # ... include other numerical features with some typical values
#         'count': 9,
#         'srv_count': 9,
#         'serror_rate': 0.00,
#         'srv_serror_rate': 0.00,
#         'rerror_rate': 0.00,
#         'srv_rerror_rate': 0.00,
#         'same_srv_rate': 1.00,
#         'diff_srv_rate': 0.00,
#         'srv_diff_host_rate': 0.00,
#         'dst_host_count': 9,
#         'dst_host_srv_count': 9,
#         'dst_host_same_srv_rate': 1.00,
#         'dst_host_diff_srv_rate': 0.00,
#         'dst_host_same_src_port_rate': 1.00,
#         'dst_host_srv_diff_host_rate': 0.00,
#         'dst_host_serror_rate': 0.00,
#         'dst_host_srv_serror_rate': 0.00,
#         'dst_host_rerror_rate': 0.00,
#         'dst_host_srv_rerror_rate': 0.00
#     },
#     'Flooding': {
#         'duration': 0,
#         'protocol_type': 'udp',
#         'service': 'domain_u',
#         'flag': 'SF',
#         'src_bytes': 0,
#         'dst_bytes': 0,
#         'land': 0,
#         'wrong_fragment': 0,
#         'urgent': 0,
#         'count': 255,
#         'srv_count': 255,
#         'serror_rate': 0.00,
#         'srv_serror_rate': 0.00,
#         'rerror_rate': 0.00,
#         'srv_rerror_rate': 0.00,
#         'same_srv_rate': 0.00,
#         'diff_srv_rate': 0.00,
#         'srv_diff_host_rate': 0.00,
#         'dst_host_count': 255,
#         'dst_host_srv_count': 255,
#         'dst_host_same_srv_rate': 0.00,
#         'dst_host_diff_srv_rate': 0.00,
#         'dst_host_same_src_port_rate': 0.00,
#         'dst_host_srv_diff_host_rate': 0.00,
#         'dst_host_serror_rate': 0.00,
#         'dst_host_srv_serror_rate': 0.00,
#         'dst_host_rerror_rate': 0.00,
#         'dst_host_srv_rerror_rate': 0.00
#     },
#     'Grayhole': {
#         'duration': 0,
#         'protocol_type': 'tcp',
#         'service': 'ftp_data',
#         'flag': 'S0',
#         'src_bytes': 0,
#         'dst_bytes': 0,
#         'land': 0,
#         'wrong_fragment': 0,
#         'urgent': 0,
#         'count': 1,
#         'srv_count': 1,
#         'serror_rate': 1.00,
#         'srv_serror_rate': 1.00,
#         'rerror_rate': 0.00,
#         'srv_rerror_rate': 0.00,
#         'same_srv_rate': 0.00,
#         'diff_srv_rate': 0.00,
#         'srv_diff_host_rate': 0.00,
#         'dst_host_count': 1,
#         'dst_host_srv_count': 1,
#         'dst_host_same_srv_rate': 0.00,
#         'dst_host_diff_srv_rate': 0.00,
#         'dst_host_same_src_port_rate': 0.00,
#         'dst_host_srv_diff_host_rate': 0.00,
#         'dst_host_serror_rate': 1.00,
#         'dst_host_srv_serror_rate': 1.00,
#         'dst_host_rerror_rate': 0.00,
#         'dst_host_srv_rerror_rate': 0.00
#     },
#     'Scheduling': {
#         'duration': 0,
#         'protocol_type': 'tcp',
#         'service': 'smtp',
#         'flag': 'REJ',
#         'src_bytes': 0,
#         'dst_bytes': 0,
#         'land': 0,
#         'wrong_fragment': 0,
#         'urgent': 0,
#         'count': 10,
#         'srv_count': 10,
#         'serror_rate': 0.30,
#         'srv_serror_rate': 0.30,
#         'rerror_rate': 0.10,
#         'srv_rerror_rate': 0.10,
#         'same_srv_rate': 0.60,
#         'diff_srv_rate': 0.20,
#         'srv_diff_host_rate': 0.10,
#         'dst_host_count': 10,
#         'dst_host_srv_count': 10,
#         'dst_host_same_srv_rate': 0.60,
#         'dst_host_diff_srv_rate': 0.20,
#         'dst_host_same_src_port_rate': 0.60,
#         'dst_host_srv_diff_host_rate': 0.10,
#         'dst_host_serror_rate': 0.30,
#         'dst_host_srv_serror_rate': 0.30,
#         'dst_host_rerror_rate': 0.10,
#         'dst_host_srv_rerror_rate': 0.10
#     },
#     'Normal': {
#         'duration': 0,
#         'protocol_type': 'icmp',
#         'service': 'ecr_i',
#         'flag': 'SF',
#         'src_bytes': 1032,
#         'dst_bytes': 0,
#         'land': 0,
#         'wrong_fragment': 0,
#         'urgent': 0,
#         'count': 1,
#         'srv_count': 1,
#         'serror_rate': 0.00,
#         'srv_serror_rate': 0.00,
#         'rerror_rate': 0.00,
#         'srv_rerror_rate': 0.00,
#         'same_srv_rate': 1.00,
#         'diff_srv_rate': 0.00,
#         'srv_diff_host_rate': 0.00,
#         'dst_host_count': 1,
#         'dst_host_srv_count': 1,
#         'dst_host_same_srv_rate': 1.00,
#         'dst_host_diff_srv_rate': 0.00,
#         'dst_host_same_src_port_rate': 1.00,
#         'dst_host_srv_diff_host_rate': 0.00,
#         'dst_host_serror_rate': 0.00,
#         'dst_host_srv_serror_rate': 0.00,
#         'dst_host_rerror_rate': 0.00,
#         'dst_host_srv_rerror_rate': 0.00
#     }
# }

# # Main app function
# def main():
#     st.title("Wireless Network Attack Detection System")

#     model, label_encoder, train_columns = load_artifacts()

#     st.write("This app uses a trained ML model to detect potential attacks based on network traffic features.")

#     attack_type = st.selectbox("🧪 Choose a test case (or select Manual Input):", ['Manual Input'] + list(sample_inputs.keys()))

#     if attack_type != 'Manual Input':
#         user_input = sample_inputs[attack_type]
#         st.success(f"✅ Auto-filled data for {attack_type}")
#     else:
#         # Manual input form for features
#         user_input = {}
#         user_input['duration'] = st.number_input("duration", min_value=0)
#         user_input['protocol_type'] = st.selectbox("protocol_type", options=['tcp', 'udp', 'icmp'])
#         user_input['service'] = st.text_input("service (e.g. http, ftp_data, domain_u, smtp, ecr_i)", value="http")
#         user_input['flag'] = st.text_input("flag (e.g. SF, S0, REJ)", value="SF")
#         user_input['src_bytes'] = st.number_input("src_bytes", min_value=0)
#         user_input['dst_bytes'] = st.number_input("dst_bytes", min_value=0)
#         user_input['land'] = st.number_input("land", min_value=0, max_value=1)
#         user_input['wrong_fragment'] = st.number_input("wrong_fragment", min_value=0)
#         user_input['urgent'] = st.number_input("urgent", min_value=0)

#         # Additional numeric features
#         numeric_features_list = [
#             'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
#             'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
#             'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
#             'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
#             'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
#             'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
#         ]

#         for feat in numeric_features_list:
#             user_input[feat] = st.number_input(feat, value=0.0 if "rate" in feat else 0)

#     if st.button("Predict Attack Type"):
#         try:
#             prediction = predict_attack(model, label_encoder, train_columns, user_input)
#             st.markdown(f"### 🛡️ Predicted Attack Type: **{prediction}**")
#         except Exception as e:
#             st.error(f"Error during prediction: {e}")

#     # For demo: show count of predicted attacks from sample inputs
#     if st.checkbox("Show prediction distribution of sample test cases"):
#         results = []
#         for name, sample in sample_inputs.items():
#             pred = predict_attack(model, label_encoder, train_columns, sample)
#             results.append(pred)

#         pred_counts = pd.Series(results).value_counts()

#         st.write("### Prediction counts on sample inputs:")
#         st.bar_chart(pred_counts)

#         fig, ax = plt.subplots()
#         pred_counts.plot.pie(autopct='%1.1f%%', ax=ax, figsize=(5,5))
#         ax.set_ylabel('')
#         st.pyplot(fig)

# if __name__ == "__main__":
#     main()
