import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Paths
DATA_PATH = os.path.join('data', 'KDDTrain+.txt')
COLUMN_NAMES_PATH = os.path.join('data', 'kdd_names.txt')
MODEL_PATH = os.path.join('models', 'random_forest_model.pkl')
ENCODER_PATH = os.path.join('models', 'label_encoder.pkl')
FEATURES_PATH = os.path.join('models', 'feature_columns.pkl')

# Load column names
print("📅 Loading column names...")
with open(COLUMN_NAMES_PATH, 'r') as f:
    lines = f.readlines()
column_names = [line.split(':')[0].strip() for line in lines[:41]]  # First 41 features
column_names.append('label')
column_names.append('difficulty_level')
  # Final column is the label

print(f"✅ Loaded {len(column_names)} columns")

# Load dataset
print("📊 Loading dataset...")
df = pd.read_csv(DATA_PATH, names=column_names, sep=',')
print("📏 DataFrame shape:", df.shape)
print("🧪 Sample label column values:", df['label'].unique()[:5])

# Clean and normalize label
print("🔄 Cleaning and standardizing label values...")
df['label'] = df['label'].astype(str).str.strip().str.lower()

# Map labels to custom attack types
print("🗂️ Mapping attack labels...")
attack_map = {
    # Black Hole
    'warezclient': 'Black Hole', 'warezmaster': 'Black Hole',
    'ftp_write': 'Black Hole', 'guess_passwd': 'Black Hole',
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
    'imap': 'Scheduling', 'multihop': 'Scheduling', 'phf': 'Scheduling',
    'spy': 'Scheduling', 'rootkit': 'Scheduling', 'loadmodule': 'Scheduling',

    # Normal
    'normal': 'Normal'
}
df['attack_type'] = df['label'].apply(lambda x: attack_map.get(x, 'Other'))

# Print sample and unknown labels
print(df.head(1))
print("Unknown labels mapped to 'Other':")
print(df[df['attack_type'] == 'Other']['label'].value_counts())

# Drop unknown attack types
df = df[df['attack_type'] != 'Other']
print(f"✅ Retained rows after filtering: {len(df)}")

# Show label distribution
print("📊 Attack type distribution before sampling:")
print(df['attack_type'].value_counts())

# Sample 10% subset
# df = df.sample(frac=0.1, random_state=42)
df = df.groupby('attack_type', group_keys=False).apply(lambda x: x.sample(frac=0.5, random_state=42))

print(f"🔁 Using 10% subset for training: {len(df)}")

# Label encode categorical features
print("🔐 Label encoding categorical features and saving encoders...")
categorical_cols = ['protocol_type', 'service', 'flag']
os.makedirs('models', exist_ok=True)

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    with open(f'models/le_{col}.pkl', 'wb') as f:
        pickle.dump(le, f)
    print(f"✅ Saved LabelEncoder for {col} at models/le_{col}.pkl")

# Clean non-numeric
non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
non_numeric_cols = [col for col in non_numeric_cols if col not in ['attack_type']]
if non_numeric_cols:
    print(f"❌ Dropping non-numeric columns: {non_numeric_cols}")
    df.drop(columns=non_numeric_cols, inplace=True)

# Handle missing/infinite
df = df.fillna(0).replace([np.inf, -np.inf], 0)

# Prepare input and target
X = df.drop(columns=['attack_type'])
y = df['attack_type']

# Save feature columns
print(f"🗕️ Saving feature columns to: {FEATURES_PATH}")
with open(FEATURES_PATH, 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

# Encode attack labels
print("🔐 Encoding attack labels...")
fixed_classes = ['Black Hole', 'Flooding', 'Grayhole', 'Normal', 'Scheduling']
label_encoder = LabelEncoder()
label_encoder.fit(fixed_classes)

if not set(y).issubset(set(fixed_classes)):
    unknown_labels = set(y) - set(fixed_classes)
    print(f"⚠️ Warning: Unknown labels found in y: {unknown_labels}")

y_encoded = label_encoder.transform(y)
print("🔍 Attack labels trained:", label_encoder.classes_)

# Save class labels
with open('models/class_labels.txt', 'w') as f:
    for cls in label_encoder.classes_:
        f.write(f"{cls}\n")

# Train/test split
print("✂️ Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train model
print("🌲 Training RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
model.fit(X_train, y_train)

# Save model and encoder
print(f"✅ Saving model → {MODEL_PATH}")
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f"✅ Saving label encoder → {ENCODER_PATH}")
with open(ENCODER_PATH, 'wb') as f:
    pickle.dump(label_encoder, f)

print("🎉 Model training complete. All artifacts saved!")
# Summary of the training process
