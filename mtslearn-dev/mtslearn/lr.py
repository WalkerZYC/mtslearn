from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X, y):
    """Train a logistic regression model on the time-series data."""
    model = LogisticRegression()
    model.fit(X, y)
    return model
