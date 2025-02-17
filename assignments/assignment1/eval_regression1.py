import numpy as np
from LinearRegression import LinearRegression

# Load the test dataset
data = np.load("iris_data.npz")
X_test, y_test = data["X_test"], data["y_test"]

# Extract features for Model 1 (Predict the Sepal Width using Sepal Length)
y_test = data["X_test"][:, 1].reshape(-1, 1)    # Sepal Width corresponds to column 1
X_test = data["X_test"][:, [0]]                 # Sepal Length corresponds to column 0

# Load trained model
model = LinearRegression()
model.load("regression1_model.npz")

# Compute Mean Squared Error (MSE) using score method (which also calculates the predicted target)
mse = model.score(X_test, y_test)

# Print and save results
print(f"Model 1 Test MSE: {mse}")
np.savez("regression1_test_results.npz", mse=mse)
