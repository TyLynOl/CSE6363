# CSE 6363 - Assignment 3: Deep Learning
# Ty Buchanan

import os
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import random_split

# DataModule for the Imagenette dataset
class ImagenetteDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4, test_split_ratio: float = 0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_split_ratio = test_split_ratio

        self.train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        full_train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=self.train_transforms)
        val_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform=self.val_transforms)

        # Split full_train_dataset into new_train and test_dataset.
        test_size = int(len(full_train_dataset) * self.test_split_ratio)
        train_size = len(full_train_dataset) - test_size
        self.train_dataset, self.test_dataset = random_split(full_train_dataset, [train_size, test_size])
        self.val_dataset = val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


# PyTorch Lightning Module for a Basic CNN
class BasicCNNModule(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        # Define the CNN architecture.
        # Assumes input images of size (3, 224, 224)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # After two pooling layers, the spatial dimensions reduce from 224x224 to 56x56.
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Adjust this if your image size or architecture changes.
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Forward pass: two conv layers followed by pooling, then fully connected layers.
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch.
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = F.cross_entropy(logits, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


if __name__ == "__main__":
    # Adjust the path to where your Imagenette data is stored.
    data_dir = "/Users/tylbuchanan/Documents/Python Projects/CSE 6363 - Machine Learning/Assignment 1 - Linear Models/CSE6363/assignments/deep_learning/imagenette2"  
    batch_size = 32
    num_workers = 4

    # Initialize the DataModule and set up datasets.
    dm = ImagenetteDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)
    dm.setup()
    
    # Test data loader by fetching one batch from the training DataLoader
    batch = next(iter(dm.train_dataloader()))
    images, labels = batch
    print("Batch images shape:", images.shape)
    print("Batch labels:", labels)


    # Determine the number of classes from the training dataset.
    if hasattr(dm.train_dataset, 'dataset'):
        num_classes = len(dm.train_dataset.dataset.classes)
    else:
        num_classes = len(dm.train_dataset.classes)

    print(f"Detected {num_classes} classes: {dm.train_dataset.dataset.classes}")

    # Initialize the Basic CNN model.
    model = BasicCNNModule(num_classes=num_classes, learning_rate=1e-3)

    # Set up early stopping callback: stops training if the validation loss does not improve for 5 epochs.
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=True, mode='min')

    # Initialize the trainer. Here, max_epochs is set to 50 (adjust as needed).
    trainer = pl.Trainer(max_epochs=50, callbacks=[early_stop_callback])

    # Train the model.
    trainer.fit(model, dm)

    # Test the model on the test set.
    trainer.test(model, datamodule=dm)
