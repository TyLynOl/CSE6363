import numpy as np
from LinearRegression import LinearRegression

# Load test dataset
data = np.load("iris_data.npz")
X_test, y_test = data["X_test"], data["y_test"]

# Extract features for Model 3 (Predict the Petal Width using Petal Length)
y_test = data["X_test"][:, 3].reshape(-1, 1)    # Petal Width corresponds to column 3
X_test = data["X_test"][:, [2]]                 # Petal Length corresponds to column 2

# Load trained model
model = LinearRegression()
model.load("regression3_model.npz")

# Compute Mean Squared Error (MSE)
mse = model.score(X_test, y_test)

# Print and save results
print(f"Model 3 Test MSE: {mse}")
np.savez("regression3_test_results.npz", mse=mse)
