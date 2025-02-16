import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 100 samples, 1 feature
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relationship with noise

# Split into training and validation sets
split_index = int(0.9 * len(X))
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train, batch_size=16, regularization=0.1, max_epochs=500, patience=10)

# Make predictions
train_predictions = model.predict(X_train)
val_predictions = model.predict(X_val)

# Plot results
plt.scatter(X_train, y_train, color='blue', label='Train Data')
plt.scatter(X_val, y_val, color='red', label='Validation Data')
plt.plot(X_train, train_predictions, color='black', linewidth=2, label='Model Prediction')
plt.xlabel("Feature X")
plt.ylabel("Target y")
plt.legend()
plt.title("Linear Regression Fit")
plt.show()

# Print final weights and bias
print("Final Weights:", model.weights)
print("Final Bias:", model.bias)
