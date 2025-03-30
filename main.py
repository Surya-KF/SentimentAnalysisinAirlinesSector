import pandas as pd
import joblib
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    return text


def train_sentiment_model(csv_file):
    df = pd.read_csv(csv_file)
    df.dropna(inplace=True)  # Handle missing values

    texts = df['Review'].astype(str).apply(preprocess_text)
    labels = df['Recommended'].map({'yes': 1, 'no': 0})

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    y = labels.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump((model, vectorizer), "sentiment_model.pkl")  # Save model
    return model, vectorizer


def predict_sentiment(model, vectorizer, new_text):
    new_text = preprocess_text(new_text)
    X_new = vectorizer.transform([new_text])
    prediction = model.predict(X_new)[0]
    return "positive" if prediction == 1 else "negative"


if __name__ == "__main__":
    csv_file = "problem2/AirlineReviews.csv"  # Update with actual file path
    model, vectorizer = train_sentiment_model(csv_file)


    while True:
        user_input = input("Enter a review (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting...")
            break
        sentiment = predict_sentiment(model, vectorizer, user_input)
        print(f"Predicted Sentiment: {sentiment}")
