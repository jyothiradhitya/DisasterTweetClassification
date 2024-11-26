import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

try:
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("TF-IDF Vectorizer loaded successfully")
except Exception as e:
    print(f"Error loading vectorizer: {e}")

try:
    model = joblib.load('logistic_regression_model.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Streamlit App
st.title("Disaster Tweets Classifier")
st.write("""
Classify tweets as disaster-related or not using machine learning.
""")

# Input form
tweet = st.text_input("Enter a Tweet", "Example: Forest fire near La Ronge Sask Canada")

if st.button("Classify"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet for classification.")
    else:
        # Preprocess and classify
        tweet_vectorized = vectorizer.transform([tweet])
        prediction = model.predict(tweet_vectorized)
        prediction_proba = model.predict_proba(tweet_vectorized)

        # Display results
        label = "Disaster-related" if prediction[0] == 1 else "Not Disaster-related"
        confidence = np.max(prediction_proba) * 100

        st.success(f"The tweet is classified as: **{label}**")
        st.info(f"Confidence: {confidence:.2f}%")

# Add visualization for dataset analysis
st.subheader("Dataset Insights")
if st.checkbox("Show Keyword Distribution"):
    # Assuming df_train is loaded
    df_train = pd.read_csv("train1.csv")  # Replace with actual path
    keyword_counts = df_train['keyword'].value_counts()[:20]
    st.bar_chart(keyword_counts)

if st.checkbox("Show Target Distribution"):
    target_counts = df_train['target'].value_counts()
    st.bar_chart(target_counts)

