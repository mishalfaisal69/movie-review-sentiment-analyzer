# 🎬 Movie Review Sentiment Analyzer

A machine learning project that predicts whether a movie review is positive or negative.

## 📌 About the Project

This project uses Natural Language Processing (NLP) and machine learning to classify movie reviews based on their sentiment.

The model was trained using 2,000 labeled movie reviews from the NLTK Movie Reviews dataset.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- NLTK
- TF-IDF
- Logistic Regression
- Gradio
- Matplotlib

## 📊 Model Performance

The model achieved:

- **82.75% accuracy**
- **0.83 macro F1-score**
- **400 test reviews**

### Confusion Matrix

The model correctly classified:

- 159 negative reviews
- 172 positive reviews

## ⚙️ How It Works

1. Movie reviews are collected from the NLTK dataset.
2. The text is cleaned and converted into numerical features using TF-IDF.
3. A Logistic Regression model is trained on the reviews.
4. The model predicts whether new reviews are positive or negative.
5. A Gradio interface allows users to test the model interactively.

## 🚀 Future Improvements

- Add neutral sentiment classification
- Improve model accuracy with a larger dataset
- Add more advanced NLP techniques
- Deploy the application online
- Improve the user interface

## 👩‍💻 Author

Mishal Faisal
