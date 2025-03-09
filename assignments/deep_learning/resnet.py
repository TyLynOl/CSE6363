# CSE 6363 - Assignment 3: Deep Learning
# Ty Buchanan

import os
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# DataModule for Imagenette dataset (using train and val folders, with a custom test split)
class ImagenetteDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4, test_split_ratio: float = 0.2):
        super().__init__()
        self.data_dir = data_dir  # expecting subfolders: train, val
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
        # Create a test split from the training data
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


# PyTorch Lightning Module for ResNet 18
class ResNet18Module(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3, pretrained=True):
        super().__init__()
        self.save_hyperparameters()
        # Load pretrained ResNet 18 model
        self.model = models.resnet18(pretrained=pretrained)
        # Replace the final fully connected layer with one that outputs 'num_classes' scores
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
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
    # Set the dataset directory path (adjust as needed)
    data_dir = "/Users/tylbuchanan/Documents/Python Projects/CSE 6363 - Machine Learning/Assignment 1 - Linear Models/CSE6363/assignments/deep_learning/imagenette2"  
    batch_size = 32
    num_workers = 4

    dm = ImagenetteDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers, test_split_ratio=0.2)
    dm.setup()

    # Check the number of classes from the underlying dataset
    num_classes = len(dm.train_dataset.dataset.classes)
    print(f"Detected {num_classes} classes: {dm.train_dataset.dataset.classes}")

    model = ResNet18Module(num_classes=num_classes, learning_rate=1e-3, pretrained=True)

    # Early stopping callback: stops training if the validation loss does not improve for 5 epochs.
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=True, mode='min')

    # Set up the Trainer
    trainer = pl.Trainer(max_epochs=50, callbacks=[early_stop_callback])
    
    # Train the model
    trainer.fit(model, dm)
    
    # Test the model
    trainer.test(model, datamodule=dm)
