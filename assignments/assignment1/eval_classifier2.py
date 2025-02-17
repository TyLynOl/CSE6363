import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from LogisticRegression import LogisticRegression

# Load test dataset
data = np.load("iris_data.npz")
X_test, y_test = data["X_test"], data["y_test"]

# Extract features for Classifier 2 (Sepal Length and Sepal Width)
X_test = X_test[:, [0, 1]]          # Sepal Length and Sepal Width

# Load all three OvR classifiers
models = []
for class_label in [0, 1, 2]:
    model = LogisticRegression()
    model.load(f"classifier2_class{class_label}.npz")
    models.append(model)

# Predict probabilities for each classifier
probabilities = np.array([model.predict(X_test, return_probs=True) for model in models]).T

# Assign the class with the highest probability
predictions = np.argmax(probabilities, axis=1)

# Compute accuracy
accuracy = np.mean(predictions == y_test)
print(f"Model 2 (Sepal Features) Test Accuracy: {accuracy:.4f}")

# Plot decision regions
for i, model in enumerate(models):
    plt.figure(figsize=(8, 6))
    plot_decision_regions(X_test, predictions, clf=model)  # Use each classifier correctly
    plt.xlabel("Sepal Length" if "classifier2" in __file__ else "Petal Length")
    plt.ylabel("Sepal Width" if "classifier2" in __file__ else "Petal Width")
    plt.title(f"Decision Boundary - Classifier for Class {i}")
    plt.savefig(f"decision_boundary_classifier{1 if 'classifier1' in __file__ else 2}_class{i}.png")
    plt.show()