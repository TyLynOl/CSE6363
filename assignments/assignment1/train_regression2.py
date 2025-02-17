import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

# Load preprocessed dataset
data = np.load("iris_data.npz")
X_train, y_train = data["X_train"], data["y_train"]

# Extract features for Model 2 (Predict Petal Length using Sepal Length and Sepal Width)
X_train = X_train[:, [0,1]]                     # Sepal Length and Sepal Width corresponds to column 0 and 1 of original dataset
y_train = data["X_train"][:, 2].reshape(-1, 1)  # Petal Length corresponds to column 2 of original dataset


# Initialize and train the model
model = LinearRegression()
train_losses = model.fit(X_train, y_train, batch_size=32, regularization=0, max_epochs=100, patience=3)

# Save the model parameters
model.save("regression2_model.npz")

# Plot training loss curve
plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
plt.xlabel("Steps (Batches)")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Training Loss Curve - Regression Model 2")
plt.legend()
plt.savefig("regression2_loss.png")             # Saves the batch vs. loss plot (for the homework report)
plt.show()

print("Training complete. Model parameters saved as 'regression2_model.npz'.")
