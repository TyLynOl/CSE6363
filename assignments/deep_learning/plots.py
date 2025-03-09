# CSE 6363 - Assignment 3: Deep Learning
# Ty Buchanan

import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV log file
csv_file = "lightning_logs/version_0/metrics.csv"
df = pd.read_csv(csv_file)

print("Full CSV head:")
print(df.head())

# Filter rows for training metrics (where train_loss is not NaN)
df_train = df[df["train_loss"].notna()]
print("Training metrics:")
print(df_train)

plt.figure(figsize=(10, 6))
plt.plot(df_train["step"], df_train["train_loss"], label="Train Loss", marker='o')
plt.xlabel("Step")
plt.ylabel("Train Loss")
plt.title("Training Loss over Steps")
plt.legend()
plt.show()

# Filter rows for validation metrics (where val_loss is not NaN)
df_val = df[df["val_loss"].notna()]
print("Validation metrics:")
print(df_val)

plt.figure(figsize=(10, 6))
plt.plot(df_val["step"], df_val["val_loss"], label="Validation Loss", marker='o')
plt.xlabel("Step")
plt.ylabel("Validation Loss")
plt.title("Validation Loss over Steps")
plt.legend()
plt.show()

# Filter rows for validation accuracy (where val_acc is not NaN)
df_val_acc = df[df["val_acc"].notna()]
plt.figure(figsize=(10, 6))
plt.plot(df_val_acc["step"], df_val_acc["val_acc"], label="Validation Accuracy", marker='o')
plt.xlabel("Step")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy over Steps")
plt.legend()
plt.show()
