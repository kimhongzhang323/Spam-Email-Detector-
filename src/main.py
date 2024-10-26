"""
spam_email_detector/src/main.py

This script is the entry point for the Spam Email Detector application. It
loads a dataset of emails, preprocesses the data, trains a machine learning
model, evaluates its performance, and tests the model with new messages.

Modules:
    - preprocess: Contains functions for loading and preprocessing the email data.
    - model: Contains functions for training the machine learning model and making predictions.
"""

from preprocess import load_data, preprocess_data, split_data, vectorize_data
from model import train_model, evaluate_model, predict_spam

def main():
    """
    Main function to execute the spam email detection workflow.
    
    Steps performed:
        1. Load the email dataset from a CSV file.
        2. Preprocess the data to extract features and labels.
        3. Split the data into training and testing sets.
        4. Vectorize the text data for model training.
        5. Train a Naive Bayes classifier on the training data.
        6. Evaluate the trained model on the testing data.
        7. Test the model with a sample email message to check if it's spam or not.
    """
    # Load and preprocess data
    data = load_data('data/spam.csv')  # Adjust the path to your dataset as needed
    X, y = preprocess_data(data)         # Extract features (X) and labels (y)
    X_train, X_test, y_train, y_test = split_data(X, y)  # Split the data into training and testing sets
    X_train_vectorized, X_test_vectorized, vectorizer = vectorize_data(X_train, X_test)  # Vectorize the text data

    # Train the model
    model = train_model(X_train_vectorized, y_train)  # Train a Naive Bayes classifier

    # Evaluate the model
    evaluate_model(model, X_test_vectorized, y_test)  # Evaluate model performance on test data

    # Test with a new message
    test_message = input("Enter a message to test: ")  # Prompt user to enter a message
    print(predict_spam(model, vectorizer, test_message))  # Check if the sample message is spam or not

if __name__ == "__main__":
    main()  # Run the main function when the script is executed
