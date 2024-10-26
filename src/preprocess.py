"""
spam_email_detector/src/preprocess.py

This module contains functions for loading and preprocessing email data for spam detection.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple, Any

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load email data from a CSV file.

    Args:
        filepath: The path to the CSV file containing the email data.

    Returns:
        A pandas DataFrame containing the email messages and their corresponding categories.
    """
    data = pd.read_csv(filepath)  # Read CSV file into a DataFrame
    return data  # Return the loaded DataFrame

def preprocess_data(data: pd.DataFrame) -> Tuple[Any, Any]:
    """
    Preprocess the email data to extract features and labels.

    Args:
        data: A pandas DataFrame containing the email messages and categories.

    Returns:
        A tuple containing:
            - X: The email messages (features).
            - y: The corresponding labels (0 for ham, 1 for spam).
    """
    X = data['Message']  # Extract email messages
    y = data['Category'].map({'ham': 0, 'spam': 1})  # Map categories to numerical values
    return X, y  # Return features and labels

def split_data(X: Any, y: Any) -> Tuple[Any, Any, Any, Any]:
    """
    Split the dataset into training and testing sets.

    Args:
        X: The features (email messages).
        y: The labels (0 for ham, 1 for spam).

    Returns:
        A tuple containing:
            - X_train: Training features.
            - X_test: Testing features.
            - y_train: Training labels.
            - y_test: Testing labels.
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)  # Split the data

def vectorize_data(X_train: Any, X_test: Any) -> Tuple[Any, Any, CountVectorizer]:
    """
    Vectorize the email messages using CountVectorizer.

    Args:
        X_train: The training email messages.
        X_test: The testing email messages.

    Returns:
        A tuple containing:
            - X_train_vectorized: Vectorized training features.
            - X_test_vectorized: Vectorized testing features.
            - vectorizer: The fitted CountVectorizer instance.
    """
    vectorizer = CountVectorizer()  # Initialize CountVectorizer
    X_train_vectorized = vectorizer.fit_transform(X_train)  # Fit and transform training data
    X_test_vectorized = vectorizer.transform(X_test)  # Transform testing data
    return X_train_vectorized, X_test_vectorized, vectorizer  # Return vectorized data and vectorizer
