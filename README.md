# Airline Customer Sentiment Analysis

## Problem 2: Binary Classification for Customer Sentiment

This project implements a machine learning solution to classify airline customer feedback as either positive or negative. The system analyzes text reviews to determine customer sentiment using natural language processing and binary classification techniques.

## Overview

The sentiment analysis system:
- Preprocesses text data from airline customer reviews
- Trains a Logistic Regression model to classify sentiment
- Provides a prediction interface for new customer feedback
- Evaluates model performance using accuracy metrics and confusion matrix

## Technical Requirements

- Python 3.8+
- scikit-learn
- pandas
- joblib
- re (Regular Expressions)

## Dataset

The system uses the `AirlineReviews.csv` dataset containing:
- **Review**: Text feedback from airline customers
- **Recommended**: Binary label ('yes'/'no') indicating positive/negative sentiment

## Project Structure

```
problem2/
├── main.py              # Main implementation file
├── AirlineReviews.csv   # Training dataset
├── sentiment_model.pkl  # Saved model and vectorizer
├── requirements.txt     # Project dependencies
└── README.md            # This documentation
```

## Implementation Details

### Data Preprocessing (`preprocess_text` function)
- Text conversion to lowercase
- Removal of punctuation
- Handling of missing values

### Model Training (`train_sentiment_model` function)
- Loads and preprocesses the dataset
- Converts labels ('yes'/'no') to binary format (1/0)
- Creates TF-IDF feature vectors from text
- Splits data into training and testing sets
- Trains a Logistic Regression classifier
- Evaluates model performance (accuracy and confusion matrix)
- Saves the trained model and vectorizer for future use

### Sentiment Prediction (`predict_sentiment` function)
- Accepts new text input
- Applies the same preprocessing used during training
- Transforms text using the saved TF-IDF vectorizer
- Returns sentiment prediction as "positive" or "negative"

## Usage Instructions

### Installation

1. Clone the repository:
```

```bash
git clone <repository-url>
```

```bash
cd problem2
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Training the Model

The model is trained automatically when running the main script:

```bash
python main.py
```

This will:
- Load the dataset from `AirlineReviews.csv`
- Preprocess the text data
- Train the sentiment classification model
- Display accuracy metrics and confusion matrix
- Save the model to `sentiment_model.pkl`

### Making Predictions

After training, the script enters an interactive mode where you can input new reviews:

```
Enter a review (or type 'exit' to quit): The flight was comfortable and the staff was friendly.
Predicted Sentiment: positive

Enter a review (or type 'exit' to quit): My luggage was lost and customer service was unhelpful.
Predicted Sentiment: negative

Enter a review (or type 'exit' to quit): exit
Exiting...
```

## Evaluation Metrics

The system evaluates the model using:
- **Accuracy**: Percentage of correctly classified reviews
- **Confusion Matrix**: Visualization of true vs. predicted classifications



