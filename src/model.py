"""
spam_email_detector/src/model.py

This module contains functions for training a Naive Bayes model for spam detection,
evaluating its performance, and predicting whether a given message is spam or not.
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from typing import Any, Tuple

def train_model(X_train_vectorized: Any, y_train: Any) -> MultinomialNB:
    """
    Train a Naive Bayes model using the provided training data.

    Args:
        X_train_vectorized: The vectorized training features.
        y_train: The labels for the training data (0 for ham, 1 for spam).

    Returns:
        A trained MultinomialNB model.
    """
    model = MultinomialNB()  # Initialize the Naive Bayes model
    model.fit(X_train_vectorized, y_train)  # Train the model on the vectorized data
    return model  # Return the trained model

def evaluate_model(model: MultinomialNB, X_test_vectorized: Any, y_test: Any) -> Any:
    """
    Evaluate the trained model using the testing data.

    Args:
        model: The trained Naive Bayes model.
        X_test_vectorized: The vectorized testing features.
        y_test: The true labels for the testing data.

    Prints:
        Accuracy and a classification report of the model's performance.
    """
    y_pred = model.predict(X_test_vectorized)  # Make predictions on the test set
    print("Accuracy:", accuracy_score(y_test, y_pred))  # Print the accuracy
    print(classification_report(y_test, y_pred))  # Print detailed classification metrics
    return y_pred  # Return the predictions

def predict_spam(model: MultinomialNB, vectorizer: Any, message: str) -> str:
    """
    Predict whether a given message is spam or not.

    Args:
        model: The trained Naive Bayes model.
        vectorizer: The CountVectorizer used for text vectorization.
        message: The email message to be classified.

    Returns:
        'Spam' if the message is classified as spam, otherwise 'Not Spam'.
    """
    message_vectorized = vectorizer.transform([message])  # Vectorize the input message
    prediction = model.predict(message_vectorized)  # Make the prediction
    return 'Spam' if prediction[0] == 1 else 'Not Spam'  # Return the classification result
