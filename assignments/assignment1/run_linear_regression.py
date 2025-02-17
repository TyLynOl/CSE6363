import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

# Generate synthetic dataset
np.random.seed()
X_train = 2 * np.random.rand(100, 1)
y_train = 4 + 3 * X_train + np.random.randn(100, 1)  # Linear relation with some noise

# Instantiate and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Generate new test data
X_test = np.array([[0], [1], [2]])  # Simple input values
y_test = 4 + 3 * X_test
y_pred = model.predict(X_test)  # Get predictions

# Print predictions
print("Predictions for X_test:")
for i in range(len(X_test)):
    print(f"X = {X_test[i][0]:.2f}, Predicted y = {y_pred[i][0]:.2f}")

# Plot predictions
plt.scatter(X_train, y_train, color="blue", label="Training Data")
plt.plot(X_test, y_pred, "r-", linewidth=2, label="Predictions")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression Predictions")
plt.show()

# Print final weights and bias
print("Final Weights:", model.weights)
print("Final Bias:", model.bias)

# Evaluate model performance
mse = model.score(X_train, y_train)
print(f"Training MSE: {mse}")

mse_val = model.score(X_test, y_test)
print(f"Test MSE: {mse_val}")

# Save model parameters
model.save("linear_model.npz")

# Load model parameters into a new instance
new_model = LinearRegression()
new_model.load("linear_model.npz")

# Verify loaded parameters match original ones
assert np.allclose(model.weights, new_model.weights), "Loaded weights do not match!"
assert np.allclose(model.bias, new_model.bias), "Loaded bias does not match!"
print("Model parameters successfully saved and loaded!")