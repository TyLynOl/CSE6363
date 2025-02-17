import numpy as np
from LogisticRegression import LogisticRegression

# Load preprocessed dataset
data = np.load("iris_data.npz")
X_train, y_train = data["X_train"], data["y_train"]

# Extract features for Classifier 2 (Sepal Length and Sepal Width)
X_train = X_train[:, [0, 1]]        # Sepal Length and Sepal Width

# Train a separate OvR classifier for each class
classes = [0, 1, 2]  # Setosa, Versicolor, Virginica
for class_label in classes:
    y_train_binary = (y_train == class_label).astype(int)  # Convert to binary (OvR)
    
    # Initialize and train the model
    model = LogisticRegression()
    model.fit(X_train, y_train_binary)
    
    # Save model parameters
    model.save(f"classifier2_class{class_label}.npz")
    print(f"Training complete. Model for class {class_label} saved as 'classifier2_class{class_label}.npz'.")
