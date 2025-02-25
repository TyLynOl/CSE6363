import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import your neural network library classes.
# Adjust the import statement if your file names or class names are different.
from layers import Layer, LinearLayer, ReLU, Sigmoid, Tanh, BinaryCrossEntropyLoss, Sequential

# ----------------------------
# Data Loading
# ----------------------------
dataset = np.load("nyc_taxi_data.npy", allow_pickle=True).item()
X_train, y_train, X_test, y_test = dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"]

print("X_train shape:", X_train.shape)
print("Number of training samples:", X_train.shape[0])
