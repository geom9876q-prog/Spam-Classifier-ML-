# Spam-Classifier-ML-
SMS/Email spam classification using TF-IDF and Logistic Regression, with model evaluation and a Streamlit-based prediction app.


# 📩 SMS / Email Spam Classification

## 📌 Overview
This project implements an **end-to-end SMS/Email spam classification system** using classical NLP and Machine Learning techniques.

The focus of the project is not only prediction, but also:
- handling **class imbalance**
- choosing **appropriate evaluation metrics**
- comparing multiple models
- selecting and deploying a **final reliable model**

A **Streamlit web application** is included for real-time spam prediction.

---

## ❓ Problem Statement
Spam detection is a **binary classification problem** where:
- the dataset is **imbalanced** (ham ≫ spam)
- different errors have **different real-world costs**

Because of this, **accuracy alone is misleading**.  
This project emphasizes **precision and recall** instead.

---

## 📊 Dataset
- **SMS Spam Collection Dataset (UCI)**
- Labels:
  - `spam`
  - `ham`
- ~5,500 messages
- Highly imbalanced dataset

---

## 🧹 Text Preprocessing
The following preprocessing steps were applied:
- Removal of non-alphanumeric characters
- Conversion to lowercase
- Tokenization
- Stemming
- **Stopwords were intentionally NOT removed** to preserve semantic meaning (e.g., negations like *not*, *don’t*)

---

## 🧠 Feature Engineering
Two approaches were explored:

### 1️⃣ Bag of Words (BoW)
- Represents text using word frequency counts
- Simple and effective baseline
- Can be noisy for very common words

### 2️⃣ TF-IDF (Term Frequency – Inverse Document Frequency)
- Weighs words based on importance
- Down-weights common but uninformative words
- Performs well without explicit stopword removal

---

## 🤖 Models Evaluated

### Naive Bayes (BoW)
- High recall
- Low precision
- Aggressive spam detection

### Logistic Regression (BoW)
- Improved precision
- Better balance compared to Naive Bayes

### CatBoost (BoW) *(Experimental)*
- Very high precision
- Lower recall
- Demonstrates limitations of tree-based models on sparse text data

---

## ✅ Final Model Selection
**Logistic Regression with TF-IDF** was selected as the final model due to:
- **Precision = 1.00** (no false positives)
- **Recall ≈ 0.88**
- Simplicity and interpretability
- Suitability for sparse, high-dimensional text features

This model provides the **best precision–recall trade-off** for real-world usage.

---

## 📈 Why Precision & Recall over Accuracy
Due to class imbalance:
- A model predicting everything as `ham` can still achieve high accuracy
- Precision and recall better represent real-world error costs

- **False Positives** → important messages marked as spam  
- **False Negatives** → spam reaching inbox  

Metric selection was driven by application context.

---

## 🚀 Web Application
A **Streamlit web application** is included for real-time prediction.

### App Features
- User enters an SMS/email
- Predicts **Spam / Not Spam**
- Displays prediction confidence
- Uses only the final selected model

📂 App files are located in the `app/` directory.

## ▶️ How to Run the App Locally
```bash
pip install -r requirements.txt
cd app
streamlit run app.py

Then open:

http://localhost:8501

🧪 Key Learnings

- Linear models can outperform complex models for text classification
- Metric choice matters more than raw accuracy
- TF-IDF improves feature quality without heavy preprocessing
- Deployment introduces real dependency and environment challenges


## 🔮 Future Improvements

Threshold tuning for recall–precision trade-off

Online deployment

Support for longer email-style messages

## 🛠️ Technologies Used

Python

Pandas, NumPy

Scikit-learn

Streamlit

## ✅ Final Note

This project demonstrates both machine learning understanding and practical deployment skills, with a focus on correctness, clarity, and real-world relevance.
