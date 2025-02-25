import numpy as np
import pandas as pd

# Set pandas options for full display.
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)
pd.set_option("display.width", 1000)

# Load the dataset.
dataset = np.load("nyc_taxi_data.npy", allow_pickle=True).item()
X_train_full = dataset["X_train"]
y_train_full = dataset["y_train"]

# Print a detailed summary of the training set.
print("=== X_train_full Head ===")
print(X_train_full.head())

print("\n=== X_train_full Info ===")
X_train_full.info()

print("\n=== X_train_full Data Types ===")
print(X_train_full.dtypes)

print("\n=== X_train_full Description ===")
print(X_train_full.describe(include='all'))

