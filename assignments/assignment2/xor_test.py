# Ty Buchanan
# CSE 6363 - 220: Assignment 2
import numpy as np
from layers import *

# Create the XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([[0], [1], [1], [0]])

# Define a simple training function
def train(model, X, y, learning_rate=0.1, epochs=10000):
    loss_fn = BinaryCrossEntropyLoss()
    
    for epoch in range(epochs):
        # Forward pass: compute predictions
        predictions = model.forward(X)
        loss = loss_fn.forward(predictions, y)
        
        # Backward pass: compute gradients
        grad_loss = loss_fn.backward()  # For a scalar loss, grad_output is 1 by default
        model.backward(grad_loss)
        
        # Update weights in layers that have trainable parameters
        for layer in model.layers:
            if hasattr(layer, "w"):
                layer.w -= learning_rate * layer.grad_w
                layer.b -= learning_rate * layer.grad_b
        
        # Optionally, print the loss every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")
    
    # After training, print final predictions
    final_preds = model.forward(X)
    print("\nFinal predictions:")
    print(final_preds)
    return model

# ------------------------------
# Model 1: Using Sigmoid activation in the hidden layer
# ------------------------------
print("Training model with Sigmoid activation in the hidden layer:")
model_sigmoid = Sequential()
model_sigmoid.add(LinearLayer(in_features=2, out_features=2))
model_sigmoid.add(Sigmoid())
model_sigmoid.add(LinearLayer(in_features=2, out_features=1))
model_sigmoid.add(Sigmoid())

trained_sigmoid = train(model_sigmoid, X, y, learning_rate=0.1, epochs=10000)

# ------------------------------
# Model 2: Using Tanh activation in the hidden layer (output layer remains sigmoid)
# ------------------------------
print("\nTraining model with Tanh activation in the hidden layer:")
model_tanh = Sequential()
model_tanh.add(LinearLayer(in_features=2, out_features=2))
model_tanh.add(Tanh())
model_tanh.add(LinearLayer(in_features=2, out_features=1))
model_tanh.add(Sigmoid())

trained_tanh = train(model_tanh, X, y, learning_rate=0.1, epochs=10000)

# ------------------------------
# Save the weights from the model that solved XOR.
# ANSWER: the sigmoid model produced the most accurate results for the XOR problem.
# ------------------------------
trained_sigmoid.save_weights("XOR_solved.w")
print("\nWeights saved to 'XOR_solved.w'")
