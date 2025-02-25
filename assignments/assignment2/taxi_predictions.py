print("Script started.")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from layers import LinearLayer, ReLU, Sigmoid, Tanh, Sequential

# ----------------------------
# Preprocessing: select num_samples random samples and normalize features.
# Assumption: X is a pandas DataFrame with at least these columns.

def preprocess_data(X):
    """
    Preprocess the dataset features by:
      - Selecting specific columns.
      - Converting datetime strings to datetime objects.
      - Extracting numeric features (month, day, hour) from the datetimes.
      - Dropping the original datetime columns.
      - Normalizing the numeric features.
    """
    # Select only the desired columns.
    columns_to_use = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude',
                      'dropoff_datetime', 'dropoff_longitude', 'dropoff_latitude']
    X = X[columns_to_use].copy()
    
    # Convert datetime columns.
    # Strip extra whitespace and convert.
    X['pickup_datetime'] = pd.to_datetime(
        X['pickup_datetime'].str.strip(), format="%Y-%m-%d %H:%M:%S", errors='coerce'
    )
    X['dropoff_datetime'] = pd.to_datetime(
        X['dropoff_datetime'].str.strip(), format="%Y-%m-%d %H:%M:%S", errors='coerce'
    )
    
    # Debug: Check dtype after conversion.
    print("After conversion, pickup_datetime dtype:", X['pickup_datetime'].dtype)
    print("After conversion, dropoff_datetime dtype:", X['dropoff_datetime'].dtype)
    
    # Drop rows where conversion failed.
    X = X.dropna(subset=['pickup_datetime', 'dropoff_datetime'])
    
    # Now extract datetime components.
    X['pickup_month'] = X['pickup_datetime'].dt.month
    X['pickup_day']   = X['pickup_datetime'].dt.day
    X['pickup_hour']  = X['pickup_datetime'].dt.hour
    
    X['dropoff_month'] = X['dropoff_datetime'].dt.month
    X['dropoff_day']   = X['dropoff_datetime'].dt.day
    X['dropoff_hour']  = X['dropoff_datetime'].dt.hour
    
    # Drop the original datetime columns.
    X = X.drop(columns=['pickup_datetime', 'dropoff_datetime'])
    
    # Normalize numeric features: zero mean, unit variance.
    mean = X.mean()
    std = X.std()
    X_norm = (X - mean) / (std + 1e-8)
    return X_norm

# ----------------------------
# Load dataset.
print("About to load dataset...")
dataset = np.load("nyc_taxi_data.npy", allow_pickle=True).item()
print("Dataset loaded.")
X_train_full, y_train_full = dataset["X_train"], dataset["y_train"]
X_test_full, y_test_full   = dataset["X_test"], dataset["y_test"]

# ----------------------------
# Select the number of random samples for training and test sets.
num_samples = 1000
indices_train = np.random.choice(X_train_full.shape[0], num_samples, replace=False)
indices_test = np.random.choice(X_test_full.shape[0], num_samples, replace=False)

X_train_subset = X_train_full.iloc[indices_train].copy() if hasattr(X_train_full, "iloc") else X_train_full[indices_train]
y_train_subset = y_train_full.iloc[indices_train].copy() if hasattr(y_train_full, "iloc") else y_train_full[indices_train]

X_test_subset = X_test_full.iloc[indices_test].copy() if hasattr(X_test_full, "iloc") else X_test_full[indices_test]
y_test_subset = y_test_full.iloc[indices_test].copy() if hasattr(y_test_full, "iloc") else y_test_full[indices_test]

# ----------------------------
# Preprocess the subsets.
X_train_proc = preprocess_data(X_train_subset)
X_test_proc  = preprocess_data(X_test_subset)

# ----------------------------
# Split the training subset into training and validation sets (80/20 split).
X_train, X_val, y_train, y_val = train_test_split(X_train_proc, y_train_subset, test_size=0.2, random_state=42)

print("Preprocessing complete.")
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test_proc.shape)

# ----------------------------
# Model builder: a network composed of (possibly) multiple hidden layers.
def build_model(config, input_dim):
    """
    Build a model based on a configuration.
    
    Parameters:
      - config: A dictionary with keys:
          'layers': a list of integers, one per hidden layer.
          'activation': the activation function to use ('relu', 'tanh', or 'sigmoid').
      - input_dim: the number of input features.
    
    Returns:
      A Sequential model with the specified layers and activation functions,
      and a final output layer of size 1.
    """
    model = Sequential()
    current_dim = input_dim
    
    # Add hidden layers.
    for nodes in config['layers']:
        model.add(LinearLayer(current_dim, nodes))
        if config['activation'] == 'relu':
            model.add(ReLU())
        elif config['activation'] == 'tanh':
            model.add(Tanh())
        else:
            model.add(Sigmoid())
        current_dim = nodes
    
    # Add final output layer for regression (1 node).
    model.add(LinearLayer(current_dim, 1))
    return model


# ----------------------------
# Loss functions: MSE and its derivative.
def mse_loss(pred, y):
    # Ensure predictions and targets are 2D arrays.
    pred = np.asarray(pred).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    return np.mean((pred - y) ** 2)

def mse_loss_deriv(pred, y):
    # Ensure predictions and targets are 2D arrays.
    pred = np.asarray(pred).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    
    # Compute derivative; result will have shape (n,1)
    return 2 * (pred - y) / y.size


# ----------------------------
# Training loop with early stopping.
def train_model(model, X_train, y_train, X_val, y_val, epochs=50, lr=0.001, early_stop=3):
    train_losses = []
    val_losses = []
    best_val = float('inf')
    stop_counter = 0
    for epoch in range(epochs):
        print(f"Starting epoch {epoch}")
        
        # Forward pass (training)
        pred_train = model.forward(X_train)
        print("Prediction shape:", np.array(pred_train).shape)
        loss_train = mse_loss(pred_train, y_train)
        train_losses.append(loss_train)
        
        # Backward pass and update
        grad = mse_loss_deriv(pred_train, y_train)
        print("Gradient shape:", np.array(grad).shape)
        model.backward(grad)
        for layer in model.layers:
            if hasattr(layer, "w"):
                layer.w -= lr * layer.grad_w
                layer.b -= lr * layer.grad_b
                
        # Validation loss
        pred_val = model.forward(X_val)
        loss_val = mse_loss(pred_val, y_val)
        val_losses.append(loss_val)
        print(f"Finished epoch {epoch}")
        print(f"Epoch {epoch}: Train Loss = {loss_train:.6f}, Val Loss = {loss_val:.6f}")
        
        # Early stopping check
        if loss_val < best_val:
            best_val = loss_val
            stop_counter = 0
        else:
            stop_counter += 1
            if stop_counter >= early_stop:
                print("Early stopping triggered.")
                break
    return train_losses, val_losses

# ----------------------------
# Define 3 hyperparameter configurations with multiple layers.
configs = [
    {'layers': [32, 16], 'activation': 'relu'},
    {'layers': [32, 16], 'activation': 'tanh'},
    {'layers': [32], 'activation': ''}
]


results = {}


input_dim = X_train.shape[1]
for i, config in enumerate(configs):
    print(f"\nTraining configuration {i+1}: {config}")
    model = build_model(config, input_dim)
    train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val, epochs=20, lr=0.001, early_stop=3)
    results[i] = {'model': model, 'train_losses': train_losses, 'val_losses': val_losses}
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title(f"Config {i+1} Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

# ----------------------------
# Evaluate models on the test set.
def rmsle(pred, y):
    # Flatten both predictions and targets to 1D arrays.
    pred_flat = np.asarray(pred).flatten()
    y_flat = np.asarray(y).flatten()
    return np.sqrt(np.mean((np.log1p(pred_flat) - np.log1p(y_flat)) ** 2))

for i, res in results.items():
    model = res['model']
    pred_test = model.forward(X_test_proc)
    test_loss = mse_loss(pred_test, y_test_subset)
    test_rmsle = rmsle(pred_test, y_test_subset)
    print(f"Config {i+1}: Test MSE Loss = {test_loss:.6f}, Test RMSLE = {test_rmsle:.6f}")
