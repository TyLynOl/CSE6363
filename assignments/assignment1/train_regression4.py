import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

# Load preprocessed dataset
data = np.load("iris_data.npz")
X_train, y_train = data["X_train"], data["y_train"]

# Extract features for Model 2 (Predict Petal Length using Sepal Width & Sepal Length)
X_train = X_train[:, [2,3]]                         # Petal Length and Petal Width corresponds to columns 2 and 3 of original dataset
y_train = data["X_train"][:, 0].reshape(-1, 1)      # Sepal Length corresponds to column 0 of original dataset


# Initialize and train the model
model = LinearRegression()
train_losses = model.fit(X_train, y_train, batch_size=32, regularization=0, max_epochs=100, patience=3)

# Save the model parameters
model.save("regression4_model.npz")

# Plot training loss curve
plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
plt.xlabel("Steps (Batches)")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Training Loss Curve - Regression Model 4")
plt.legend()
plt.savefig("regression4_loss.png")                 # Saves the batch vs. loss plot (for the homework report)
plt.show()

print("Training complete. Model parameters saved as 'regression4_model.npz'.")