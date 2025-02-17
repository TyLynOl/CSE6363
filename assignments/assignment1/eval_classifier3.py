import numpy as np
from LogisticRegression import LogisticRegression

# Load test dataset
data = np.load("iris_data.npz")
X_test, y_test = data["X_test"], data["y_test"]

# Extract features for Classifer 3 (All Features: Sepal and Petal Length/Width)
X_test = X_test[:, [0, 1, 2, 3]]  # All features

# Load all three OvR classifiers
models = []
for class_label in [0, 1, 2]:
    model = LogisticRegression()
    model.load(f"classifier3_class{class_label}.npz")
    models.append(model)

# Predict probabilities for each classifier
probabilities = np.array([model.predict(X_test, return_probs=True) for model in models]).T

# Assign the class with the highest probability
predictions = np.argmax(probabilities, axis=1)

# Compute accuracy
accuracy = np.mean(predictions == y_test)
print(f"Model 3 (All Features) Test Accuracy: {accuracy:.4f}")
