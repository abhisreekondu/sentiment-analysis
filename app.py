import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load trained model and vectorizer
model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Text preprocessing
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text.lower())
    filtered = [t for t in tokens if t not in stopwords.words('english')]
    return ' '.join(filtered)

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.markdown("## ✨ Product Review Sentiment Analyzer")
st.markdown("Enter your review below and click **Predict Sentiment** to classify it as Positive, Negative, or Neutral.")

# Input field
user_input = st.text_area("📝 Your Review", height=150)

# Predict button
if st.button("🔍 Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a review to analyze.")
    else:
        cleaned = preprocess_text(user_input)
        vec_input = vectorizer.transform([cleaned])
        prediction = model.predict(vec_input)[0]

        st.markdown("---")
        st.subheader("🔎 Review Analyzed")
        st.info(f"**\"{user_input.strip()}\"**")

        # Output based on prediction
        if prediction == 1:
            st.success("✅ This seems like a **Positive Review**.")
        elif prediction == 0 or prediction == -1:
            st.error("❌ This appears to be a **Negative Review**.")
        else:
            st.warning("⚖️ This seems to be a **Neutral Review**.")
