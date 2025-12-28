import streamlit as st
import pickle

# load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.title("📩 SMS / Email Spam Classifier")

text = st.text_area("Enter a message")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter a message.")
    else:
        text_vector = vectorizer.transform([text])
        prediction = model.predict(text_vector)[0]
        prob = model.predict_proba(text_vector)[0]

        if prediction == 1:
            st.error(f"🚨 SPAM\nConfidence: {prob[1]*100:.2f}%")
        else:
            st.success(f"✅ NOT SPAM\nConfidence: {prob[0]*100:.2f}%")

st.markdown("---")
st.markdown("""
**Model:** Logistic Regression  
**Features:** TF-IDF  
**Precision:** 1.00  
**Recall:** 0.88  
""")
