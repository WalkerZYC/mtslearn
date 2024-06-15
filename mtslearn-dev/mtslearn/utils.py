from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test data."""
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)
