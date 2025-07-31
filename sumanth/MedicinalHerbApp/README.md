# 🌿 Medicinal Herbs Identification System

A lightweight web application that uses Deep Learning and LLMs to identify medicinal herbs based on leaf images, and retrieve relevant medicinal information — empowering students, herbalists, farmers, and the general public.

---

## 🚀 Features

- 🌱 **Leaf Image Classification** using a fine-tuned **ResNet50** model
- 🔍 **Real-time Medicinal Info** fetched from **Gemini Pro API**
- 📷 Simple **Streamlit Web UI** for uploading and viewing results
- 🧠 Integrated **LLM** to describe herb properties and preparation
- 📊 Accuracy target: ≥ 90% across 30 herb classes
- 📱 Responsive UI optimized for desktop and mobile

---

## 📷 Sample Workflow

1. User uploads a clear image of a medicinal leaf
2. The model predicts the herb class (e.g., Tulsi, Neem, Amla)
3. The app queries Gemini API for:
   - Medicinal uses
   - Preparation methods (e.g., teas, pastes)
   - Cultural significance
4. Output is shown in a clean, friendly UI

---

## 🧩 Modules Overview

- `app.py` – Streamlit web app interface
- `image_preprocessing.py` – Prepares images for prediction
- `model_loader.py` – Loads and runs the ResNet50 model
- `client.py` – Queries the Gemini API
- `config.yaml` – Configuration for model/API settings
- `classes.txt` – Maps class indices to herb names

---

## 🛠️ Tech Stack

| Layer     | Tech                        |
|-----------|-----------------------------|
| Language  | Python 3.10+                |
| Frontend  | Streamlit                   |
| Backend   | PyTorch, torchvision        |
| AI Model  | ResNet50 (fine-tuned)       |
| LLM       | Gemini API (via `requests`) |
| Tools     | PIL, dotenv, unittest       |

---

## 🗂️ Dataset

- 30 medicinal herb classes (e.g., Tulsi, Neem, Ashwagandha)
- ~1,000–1,500 images total
- Augmented via random flips, rotations
- Organized in `/data/<herb_name>/*.jpg`

---

## ✅ Functional Requirements

- Upload `.jpg` or `.png` leaf image (max 5MB)
- Predict herb with ≥90% accuracy
- Display name, uses, and preparation tips
- Handle bad inputs (blurry, wrong format, unknown herb)

---

## 📋 Project Milestones

- ✅ Data Collection & Preprocessing
- ✅ ResNet50 Training & Testing
- ✅ Streamlit UI Development
- ✅ LLM Integration (Gemini API)
- ✅ End-to-End Testing
- ✅ Deployment on Streamlit Cloud

---

## 🧪 Testing Strategy

| Level          | Tools      | Status      |
|----------------|------------|-------------|
| Unit Tests     | `unittest` | ✅ Done      |
| Integration    | Manual     | ✅ Covered   |
| System Testing | UAT Cases  | ✅ Verified  |
| Coverage       | ≥ 80%      | ✅ Passed    |

---

## 🔐 Security & Deployment

- `.env` used for storing sensitive API keys
- Deployed via Streamlit Cloud (HTTPS secured)
- No user data stored — privacy-first by design

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/datalegos/our_kraft.git
cd our_kraft/sumanth/MedicinalHerbApp

# Install dependencies
pip install -r requirements.txt

# Set environment variable (Gemini API key)
echo "GEMINI_API_KEY=your_api_key" > .env

# Run the app
streamlit run app.py
