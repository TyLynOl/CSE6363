# Ty Buchanan - CSE 6363 Assignment 1
# Import numpy for array operations
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Fit a linear model.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience

        # TODO: Initialize the weights and bias based on the shape of X and y.
        # First, let's obtain the shape of X to produce appropriately shaped matrices.
        
        # N should be the number of samples, D should be the number of features.
        N, D = X.shape                  # X.shape[0] = number of data samples; X.shape[1] = number of features
        
        # Initialize the weights and biases of the instance of the class (i.e. use 'self').
        # Initialize the weights from a random distribution (0,1), and scale down to prevent large gradients. 
        self.weights = np.random.randn(D, 1)*0.01
        self.bias = np.zeros((1,))

        # TODO: Implement the training loop.
        # First, ensure that the dataset is randomly shuffled prior to assigning the training and validation sets. Use the homework specified split.
        # Shuffle the data indices to prevent unintended ordering biases. That is, reduce the likelihood of overfitting on a subset.
        indices = np.arange(N)          # Number of indices should equal the number of data samples
        np.random.shuffle(indices)      # Shuffles the array of indices randomly along one axis 
        
        # Split the dataset into training and validation sets. Assignment says to use a 90-10 split.
        split_index = int(0.9 * N)                                              # The boundary index for the first 90% of data samples (e.g. if N = 10, then training set is 1-9)
        train_index, val_index = indices[:split_index], indices[split_index:]   # Extract the data subsets to training and validation arrays using the boundary index
        
        # Create the training and validation sets
        X_train, y_train = X[train_index], y[train_index]
        X_val, y_val = X[val_index], y[val_index]
        
        # Create a tracker for the best loss, model parameters, and also for tracking validation performance stagnation
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_weights = None
        best_bias = None
        
        # Arrays to store loss scores
        train_losses = []
        val_losses = []
        
        # Begin the training loop
        for epoch in range(self.max_epochs):
            # We shuffle the training data each epoch to once again prevent any ordering bias (i.e. overfitting to a subset of our subset)
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]   # Reorders the arrays according to the array of shuffled indices
            
            # Process the mini-batches
            for i in range(0, X_train.shape[0], self.batch_size):
                X_batch = X_train[i:i + self.batch_size]            # Always take a subset of data from [i: i + batch.size], since i iterates by this difference
                y_batch = y_train[i:i + self.batch_size]            # Always take a subset of data from [i: i + batch.size], since i iterates by this difference
                
                # Calculate the predictions using XW + b
                y_pred = np.dot(X_batch, self.weights) + self.bias  # Regression calculates ^Y^ with XW + b; contrary to WX + b in neural networks (deep learning)
                
                # Calculate the loss using Mean Squared Error
                error = y_batch - y_pred                            # error = (y - ^y^)
                loss = np.mean(error**2)                            # loss = (1/N)*(error)*(error) , where N = size of subset
                
                # Track the loss each batch
                train_losses.append(loss)                           # Stores the loss for this batch (batch = step)
                
                # Calculate the gradients using provided gradient formulas
                dW = -2 * np.dot(X_batch.T, error) / X_batch.shape[0]   # Partial of L w.r.t W, use chain rule
                db = -2 * np.mean(error)
             
                ## APPLICATION OF L2 REGULARIZATION WHEN NECESSARY ##
                if self.regularization > 0:
                    dW += 2 * self.regularization * self.weights    # Look into regularization and this derivation when possible
                    
                # Perform the gradient descent update
                # ALPHA IS ASSUMED TO BE 0.01. Set alpha to be a parameter in the method header.
                self.weights -= 0.01 * dW                           # NOTE: alpha is assumed to be 0.01 here. Parameter can be passed / set in method header too.
                self.bias -= 0.01 * db                              # NOTE: alpha is assumed to be 0.01 here. Parameter can be passed / set in method header too.
                
            # Calculate the training loss after each epoch
            """
            * Training Error was calculated after each epoch; homework instructed to calculate after each batch.
            Moved the loss calculation inside the batch loop. *
            ---------
            train_error = y_train - np.dot(X_train, self.weights) - self.bias
            train_loss = np.mean(train_error ** 2)
            train_losses.append(train_loss)
            ---------
            """
            
            # Calculate the validation loss after each epoch
            val_error = y_val - np.dot(X_val, self.weights) - self.bias
            val_loss = np.mean(val_error ** 2)
            val_losses.append(val_loss)
            
            # Implementation of early stopping after "patience" epochs of stagnation
            if val_loss < best_val_loss:
                best_val_loss = val_loss                            # Best validation is overriden with current score
                epochs_without_improvement = 0                      # Reset the stagnation counter
                
                # Store the new best model parameters
                best_weights = self.weights.copy()
                best_bias = self.bias.copy()
            else:
                epochs_without_improvement += 1                     # Increment for each epoch during stagnation
                
            if epochs_without_improvement >= self.patience:
                print(f"Early stopping triggered at epoch {epoch} after {self.patience} epochs of stagnation.")
                break

            
        # Restore the best model parameters
        self.weights = best_weights
        self.bias = best_bias
        
        # Plot the training and validation loss curves
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()              
        
    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
            
        Returns
        ----------
        numpy.ndarray
            The predicted values.
        """
        # TODO: Implement the prediction function.
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
            
        Returns
        -------
        float
            The mean squared error (MSE).
        """
        # TODO: Implement the scoring function.
        y_prediction = self.predict(X)              # Compute predictions with test set
        loss = np.mean((y - y_prediction) ** 2)     # Compute loss using the MSE
        return loss
    
    def save(self, file_path):
        """Saves the model's parameters to a file.
        
        Parameters
        ----------
        file_path : str
            The file path where the model parameters will be saved.
        """
        
        np.savez(file_path, weights=self.weights, bias=self.bias)
        print(f"The model's parameters have been saved to {file_path}.")
        
    def load(self, file_path):
        """Loads the model's parameters from a file.
        
        Parameters
        ----------
        file_path : str
            The file path where the model parameters will be saved.
        """
        
        data = np.load(file_path)
        self.weights = data["weights"]
        self.bias = data["bias"]
        print(f"The model's parameters have been loaded from {file_path}.")