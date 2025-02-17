import numpy as np
from LinearRegression import LinearRegression

# Load test dataset
data = np.load("iris_data.npz")
X_test, y_test = data["X_test"], data["y_test"]

# Extract features for Model 4 (Predict Sepal Length using Petal Features)
y_test = data["X_test"][:, 0].reshape(-1, 1)    # Sepal Length corresponds to column 0
X_test = data["X_test"][:, [2, 3]]              # Petal Length and Petal Width correspond to columns 2 and 3

# Load trained model
model = LinearRegression()
model.load("regression4_model.npz")

# Compute Mean Squared Error (MSE)
mse = model.score(X_test, y_test)

# Print and save results
print(f"Model 4 Test MSE: {mse}")
np.savez("regression4_test_results.npz", mse=mse)
