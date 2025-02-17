# Ty Buchanan - CSE 6363 Assignment 1
import numpy as np
from scipy.special import expit  # Import the sigmoid function

class LogisticRegression:
    def __init__(self, learning_rate=0.01, batch_size=32, max_epochs=1000, tolerance=1e-4):
        """Logistic Regression using Mini-Batch Gradient Descent.
        
        Parameters:
        -----------
        learning_rate: float
            The factor of the gradient descent step
        batch_size: int
            The number of samples per batch.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        """Sigmoid activation function."""
        return expit(z)

    def fit(self, X, y):
        """Train the logistic regression model using mini-batch gradient descent.
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix (N x D)
        y : ndarray
            Target labels (N, )
        """
        N, D = X.shape
        self.weights = np.zeros(D)
        self.bias = 0

        for epoch in range(self.max_epochs):
            indices = np.arange(N)
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]  # Shuffle the data each epoch
            
            for i in range(0, N, self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]
                
                linear_output = np.dot(X_batch, self.weights) + self.bias
                predictions = self.sigmoid(linear_output)
                
                error = predictions - y_batch
                dW = np.dot(X_batch.T, error) / X_batch.shape[0]
                db = np.mean(error)
                
                self.weights -= self.learning_rate * dW
                self.bias -= self.learning_rate * db
                
            # Stop if weight updates are small
            if np.linalg.norm(dW) < self.tolerance:
                break

    def predict(self, X, return_probs=False):
        """Predict class labels for a given input data.
        
        Parameters:
        -----------
        X : ndarray
            Input feature matrix (N x D)
        return_probs : bool, optional
            If True, return probabilities instead of class labels.
       
        Returns:
        --------
        ndarray : Probabilities (if return_probs=True) or Predicted class labels (N,)
        """
        linear_output = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(linear_output)
        if return_probs:
            return probabilities    # Returning the raw probabilities
        return (probabilities >= 0.5).astype(int)
    
    def accuracy(self, X, y):
        """Compute accuracy of the model."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def save(self, file_path):
        """Save the model parameters to a file."""
        np.savez(file_path, weights=self.weights, bias=self.bias)
        print(f"Model parameters saved to {file_path}")
    
    def load(self, file_path):
        """Load the model parameters from a file."""
        data = np.load(file_path)
        self.weights = data["weights"]
        self.bias = data["bias"]
        print(f"Model parameters loaded from {file_path}")