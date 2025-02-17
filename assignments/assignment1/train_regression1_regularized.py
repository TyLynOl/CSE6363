import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

# Load preprocessed dataset
data = np.load("iris_data.npz")
X_train, y_train = data["X_train"], data["y_train"]

# Extract features for this model (Model 1: Predict Sepal Width using Sepal Length)
X_train = X_train[:, [0]]                       # Sepal Length corresponds to column 0 of original dataset
y_train = data["X_train"][:, 1].reshape(-1, 1)  # Sepal Width corresponds to column 1 of original dataset

# Load previously trained model (without regularization)
non_reg_model = LinearRegression()
non_reg_model.load("regression1_model.npz")
non_reg_weights, non_reg_bias = non_reg_model.weights, non_reg_model.bias

# Train a new model with L2 Regularization; FOR HOMEWORK, ASSUME 0.1 REGULARIZATION
reg_model = LinearRegression()
train_losses = reg_model.fit(X_train, y_train, batch_size=32, regularization=0.1, max_epochs=100, patience=3)

# Save the new model parameters
reg_model.save("regression1_model_regularized.npz")

# Compare weight differences
reg_weights, reg_bias = reg_model.weights, reg_model.bias
weight_diff = reg_weights - non_reg_weights
bias_diff = reg_bias - non_reg_bias

# Save the parameter differences (between non-regularized and regularized model)
np.savez("regression1_weight_differences.npz", weight_diff=weight_diff, bias_diff=bias_diff)

# Plot training loss curve
plt.plot(range(len(train_losses)), train_losses, label="Training Loss (Regularized)")
plt.xlabel("Steps (Batches)")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Training Loss Curve - Regression Model 1 (With L2 Regularization)")
plt.legend()
plt.savefig("regression1_loss_regularized.png") # Saves the batch vs. loss plot (for the homework report)
plt.show()

print("Regularized training complete. Model parameters saved as 'regression1_model_regularized.npz'.")
print("Weight and bias differences saved in 'regression1_weight_differences.npz'.")