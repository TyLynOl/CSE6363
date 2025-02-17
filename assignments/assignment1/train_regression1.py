import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

# Load preprocessed dataset
data = np.load("iris_data.npz")
X_train, y_train = data["X_train"], data["y_train"]

# Extract features for this model (Model 1: Predict Sepal Width using Sepal Length)
X_train = X_train[:, [0]]                       # Sepal Length corresponds to column 0 of original dataset
y_train = data["X_train"][:, 1].reshape(-1, 1)  # Sepal Width corresponds to column 1 of original dataset

# Initialize and train the model
model = LinearRegression()
train_losses = model.fit(X_train, y_train, batch_size=32, regularization=0, max_epochs=100, patience=3)

# Save the model parameters
model.save("regression1_model.npz")

# Plot training loss curve
plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
plt.xlabel("Steps (Batches)")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Training Loss Curve - Regression Model 1")
plt.legend()
plt.savefig("regression1_loss.png")             # Saves the batch vs. loss plot (for the homework report)
plt.show()

print("Training complete. Model parameters saved as 'regression1_model.npz'.")
