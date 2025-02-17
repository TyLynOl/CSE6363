import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target                       # Features (data) and labels (target) from the dataset

# Convert to DataFrame for easier handling
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y                                    # Store the data labels in a column named "targets"

# Split dataset into (according to homework): training (90%) and test (10%) while maintaining class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# Save the partitioned datasets for usage later
np.savez("iris_data.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

print("Data preparation complete. Training and test sets saved to iris_data.npz.")