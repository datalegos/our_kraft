# data_preprocess.py
import pandas as pd
import os
import pickle
from sklearn.preprocessing import LabelEncoder

RAW_DATA_PATH = "data/KDDTrain+.txt"
PROCESSED_DATA_PATH = "data/processed_train.csv"
ENCODER_PATH = "models/attack_label_encoder.pkl"
TRAIN_COLUMNS_PATH = "models/train_columns.pkl"

# Define column names (from KDD dataset docs)
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack_type", "difficulty_level"
]

attack_map = {
    # Black Hole
    'warezclient': 'Black Hole', 'warezmaster': 'Black Hole', 'ftp_write': 'Black Hole',
    'teardrop': 'Black Hole',

    # Flooding
    'apache2': 'Flooding', 'mailbomb': 'Flooding', 'ping_of_death': 'Flooding',
    'snmpgetattack': 'Flooding', 'snmpguess': 'Flooding', 'processtable': 'Flooding',
    'udpstorm': 'Flooding', 'neptune': 'Flooding', 'smurf': 'Flooding',
    'back': 'Flooding', 'pod': 'Flooding', 'land': 'Flooding',

    # Grayhole
    'ipsweep': 'Grayhole', 'mscan': 'Grayhole', 'nmap': 'Grayhole',
    'portsweep': 'Grayhole', 'saint': 'Grayhole', 'satan': 'Grayhole',

    # Scheduling
    'imap': 'Scheduling', 'multihop': 'Scheduling',
    'phf': 'Scheduling', 'spy': 'Scheduling', 'rootkit': 'Scheduling',
    'loadmodule': 'Scheduling', 'perl': 'Scheduling',

    # Normal
    'normal': 'Normal'
}


# === Load raw data ===
print("📥 Loading raw data...")
df = pd.read_csv(RAW_DATA_PATH, names=column_names)

# Normalize attack column
df["attack_type"] = df["attack_type"].astype(str).str.strip().str.lower()

# === Map attacks to categories ===
print("🔁 Mapping attack types...")
df["attack_label"] = df["attack_type"].apply(lambda x: attack_map.get(x, "Other"))

# Remove unknown attacks
df = df[df["attack_label"] != "Other"]

# Drop unused columns
df.drop(["attack_type", "difficulty_level"], axis=1, inplace=True)

# === Encode categorical columns ===
cat_cols = ["protocol_type", "service", "flag"]
cat_encoders = {}

print("🔢 Encoding categorical features...")
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    cat_encoders[col] = le

# === Encode attack labels ===
print("🔐 Encoding attack labels...")
label_encoder = LabelEncoder()
df["attack_label"] = label_encoder.fit_transform(df["attack_label"])
print("🔍 Attack Label Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))


# === Save encoders and features ===
os.makedirs("models", exist_ok=True)
with open(ENCODER_PATH, "wb") as f:
    pickle.dump(label_encoder, f)
with open("models/cat_encoders.pkl", "wb") as f:
    pickle.dump(cat_encoders, f)
with open(TRAIN_COLUMNS_PATH, "wb") as f:
    pickle.dump([c for c in df.columns if c != "attack_label"], f)

# === Save processed CSV ===
print("💾 Saving processed dataset...")
os.makedirs("data", exist_ok=True)
df.to_csv(PROCESSED_DATA_PATH, index=False)

print("\n📊 Final class distribution:")
print(df["attack_label"].value_counts())

print("✅ Data preprocessing completed.")
