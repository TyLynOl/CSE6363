import numpy as np
from LinearRegression import LinearRegression

# Load test dataset
data = np.load("iris_data.npz")
X_test, y_test = data["X_test"], data["y_test"]

# Extract features for Multi-Output Regression (Predict the Petal Length and Petal Width using Sepal Length and Sepal Width)
y_test = data["X_test"][:, [2, 3]]  # Petal Length and Petal Width (columns 2 and 3)
X_test = data["X_test"][:, [0, 1]]  # Sepal Length and Sepal Width (columns 0 and 1)

# Load trained model
model = LinearRegression()
model.load("multiregression_model.npz")

# Compute Mean Squared Error (MSE)
mse = model.score(X_test, y_test)

# Print and save results
print(f"Multi-Output Regression Test MSE: {mse}")
np.savez("multiregression_test_results.npz", mse=mse)
