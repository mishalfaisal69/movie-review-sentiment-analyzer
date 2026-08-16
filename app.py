import streamlit as st
import pandas as pd
import nltk

from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Page setup
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬"
)

st.title("🎬 Movie Review Sentiment Analyzer")
st.write(
    "Enter a movie review and the machine-learning model "
    "will predict whether it is positive or negative."
)


# Download dataset
nltk.download("movie_reviews", quiet=True)


# Load movie reviews
documents = [
    (list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]


# Prepare dataset
reviews = []

for words, sentiment in documents:
    text = " ".join(words)

    reviews.append({
        "review": text,
        "sentiment": sentiment
    })

df = pd.DataFrame(reviews)


# Split data
X = df["review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Convert text to numbers
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_vectors = vectorizer.fit_transform(X_train)


# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_vectors, y_train)


# User input
review = st.text_area(
    "Enter your review:",
    placeholder="Example: This movie was absolutely amazing!"
)


if st.button("Analyze Sentiment"):

    if review.strip():

        review_vector = vectorizer.transform([review])

        prediction = model.predict(review_vector)[0]

        probabilities = model.predict_proba(review_vector)[0]

        confidence = max(probabilities) * 100

        if prediction == "pos":
            st.success(
                f"😊 Positive — {confidence:.1f}% confidence"
            )
        else:
            st.error(
                f"😞 Negative — {confidence:.1f}% confidence"
            )

    else:
        st.warning("Please enter a review first!")
