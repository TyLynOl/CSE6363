# CSE 6363 - Assignment 3: Deep Learning
# Ty Buchanan

import os
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# DataModule with additional data augmentation for regularization
class AugmentedImagenetteDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4, test_split_ratio: float = 0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_split_ratio = test_split_ratio
        
        # Enhanced data augmentation for the training set
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Validation transforms remain standard
        self.val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        full_train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=self.train_transforms)
        val_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform=self.val_transforms)
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


# Basic CNN model remains the same as before
class BasicCNNModule(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
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
    data_dir = "/Users/tylbuchanan/Documents/Python Projects/CSE 6363 - Machine Learning/Assignment 1 - Linear Models/CSE6363/assignments/deep_learning/imagenette2"  
    batch_size = 32
    num_workers = 4

    dm_aug = AugmentedImagenetteDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)
    dm_aug.setup()

    num_classes = len(dm_aug.train_dataset.dataset.classes)
    print(f"Detected {num_classes} classes: {dm_aug.train_dataset.dataset.classes}")

    model_aug = BasicCNNModule(num_classes=num_classes, learning_rate=1e-3)

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=True, mode='min')
    trainer = pl.Trainer(max_epochs=50, callbacks=[early_stop_callback])
    
    trainer.fit(model_aug, dm_aug)
    trainer.test(model_aug, datamodule=dm_aug)
