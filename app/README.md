## Streamlit Application

This folder contains the Streamlit web application for real-time SMS/Email spam prediction.

### Contents
- `app.py` – Streamlit app script
- `spam_model.pkl` – Trained Logistic Regression model
- `tfidf_vectorizer.pkl` – TF-IDF vectorizer used for text transformation

### Model Details
- Model: Logistic Regression
- Features: TF-IDF
- Evaluation focus: Precision and Recall

### How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
