import numpy as np
from LinearRegression import LinearRegression

# Load test dataset
data = np.load("iris_data.npz")
X_test, y_test = data["X_test"], data["y_test"]

# Extract features for Model 2 (Predict the Petal Length using Sepal Width & Sepal Length)
y_test = data["X_test"][:, 2].reshape(-1, 1)    # Petal Length corresponds to column 2
X_test = data["X_test"][:, [0, 1]]              # Sepal Length and Sepal Width correspond to columns 0 and 1

# Load trained model
model = LinearRegression()
model.load("regression2_model.npz")

# Compute Mean Squared Error (MSE)
mse = model.score(X_test, y_test)

# Print and save results
print(f"Model 2 Test MSE: {mse}")
np.savez("regression2_test_results.npz", mse=mse)
