import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

# Load preprocessed dataset
data = np.load("iris_data.npz")
X_train, y_train = data["X_train"], data["y_train"]

# Extract features for Multi-Output Regression (Predict the Petal Length and Petal Width using Sepal Length and Sepal Width)
y_train = data["X_train"][:, [2, 3]]  # Petal Length and Petal Width (columns 2 and 3)
X_train = data["X_train"][:, [0, 1]]  # Sepal Length and Sepal Width (columns 0 and 1))

# Initialize and train the model
model = LinearRegression()
train_losses = model.fit(X_train, y_train, batch_size=32, regularization=0, max_epochs=100, patience=3)

# Save model parameters
model.save("multiregression_model.npz")

# Plot training loss curve
plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
plt.xlabel("Steps (Batches)")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Training Loss Curve - Multi-Output Regression")
plt.legend()
plt.savefig("multiregression_loss.png")                 # Saves the batch vs. loss plot (for the homework report)
plt.show()

print("Training complete. Model parameters saved as 'multiregression_model.npz'.")
