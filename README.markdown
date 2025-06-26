Extremist Reviewer Detection System
Overview
The Extremist Reviewer Detection System identifies and characterizes extremist reviewer groups by analyzing review texts for sentiment and toxicity, clustering users based on behavior, and visualizing results via a Streamlit dashboard. The system processes raw review data, cleans it, classifies reviews, clusters users, and generates visualizations.
Project Structure
/extremist-review-detector/
├── data/
│   ├── raw/
│   │   └── reviews.csv           # Input review data
│   └── processed/                # Cleaned, classified, and clustered data
├── models/                       # Trained model files
├── scripts/
│   ├── cleaner.py                # Text preprocessing
│   ├── classifier.py             # Sentiment and toxicity classification
│   ├── clustering.py             # User clustering
│   ├── visualizer.py             # Visualization generation
├── dashboard/
│   └── plots/                    # Output plots (PNG)
├── tests/
│   ├── test_cleaner.py           # Tests for cleaner.py
│   ├── test_classifier.py        # Tests for classifier.py
│   ├── test_clustering.py        # Tests for clustering.py
│   ├── test_visualizer.py        # Tests for visualizer.py
├── app.py                        # Streamlit dashboard
├── main.py                       # Pipeline orchestration
├── requirements.txt              # Python dependencies
├── README.markdown               # Project documentation

Prerequisites

Python 3.12.7
PowerShell (Windows) or terminal (Linux/macOS)
Virtual environment (recommended)
NLTK data (downloaded during setup)

Setup Instructions

Clone or Download the Project

Place the project in C:\Users\NAVEEN\Desktop\extremist-review-detector (Windows) or your preferred directory.


Open PowerShell (Windows)
powershell


Navigate to Project Directory
cd C:\Users\NAVEEN\Desktop\extremist-review-detector


Deactivate Anaconda (if active)
conda deactivate

Repeat until no environment is active (no (base) in prompt).

Create and Activate Virtual Environment
python -m venv venv
.\venv\Scripts\Activate.ps1


Upgrade pip
python -m pip install --upgrade pip


Install Dependencies
pip install -r requirements.txt


Download NLTK Resources
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"



Usage

Prepare Input Data

Ensure data/raw/reviews.csv exists with columns: user_id, review_text, timestamp.
Example:user_id,review_text,timestamp
1,Great product! Love it!,2023-01-01
2,Terrible, I HATE it <b>so much</b> 😡,2023-01-02




Run Tests
pytest tests/

Expected: All 10 tests pass.

Run Pipeline
python main.py

Outputs:

data/processed/cleaned_reviews.csv
data/processed/classified_reviews.csv
data/processed/clustered_users.csv
dashboard/plots/*.png (sentiment, toxicity, clusters, word cloud)


Run Dashboard
streamlit run app.py


Access at http://localhost:8501.
Alternatively:Start-Process "http://localhost:8501"





Features

Text Cleaning: Removes HTML tags, converts emojis, handles contractions, and lemmatizes text.
Sentiment Analysis: Classifies reviews as Positive, Negative, or Neutral using TextBlob.
Toxicity Classification: Identifies toxic reviews using a logistic regression model or rule-based approach.
User Clustering: Groups users based on toxicity, review count, and negative sentiment ratio.
Visualizations: Includes sentiment distribution, toxicity counts, user clusters, and toxic word cloud.
Dashboard: Interactive Streamlit interface for exploring results.

Troubleshooting

Tests Fail: Verify reviews.csv exists, check pytest output, and ensure dependencies are installed.
Pipeline Errors: Check logs from python main.py and confirm input data format.
Dashboard Issues: Ensure data/processed/ and dashboard/plots/ contain required files.
ModuleNotFoundError: Confirm test files include sys.path.append for scripts/ imports.
NLTK Errors: Re-run NLTK download command.
Anaconda Conflicts: Deactivate Anaconda and use the virtual environment.

License
MIT License. See LICENSE file for details (not included in this repository).
Contact
For issues, contact the project maintainer via GitHub issues or email (placeholder@example.com).
