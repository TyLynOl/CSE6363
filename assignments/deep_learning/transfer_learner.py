# CSE 6363 - Assignment 3: Deep Learning
# Ty Buchanan

import os
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# DataModule for the CIFAR10 dataset
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 128, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Training transforms with augmentation for CIFAR10
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ])
        # Test/Validation transforms
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ])

    def prepare_data(self):
        # Download CIFAR10 if not already available
        datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.cifar10_train = datasets.CIFAR10(root=self.data_dir, train=True, transform=self.transform_train)
        self.cifar10_val = datasets.CIFAR10(root=self.data_dir, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

# Lightning Module for ResNet18 Transfer Learning
class ResNet18TransferModule(pl.LightningModule):
    def __init__(self, num_classes: int = 10, learning_rate: float = 1e-3, 
                 use_pretrained_imagenette: bool = False, imagenette_checkpoint_path: str = None):
        super().__init__()
        self.save_hyperparameters()
        # Initialize a ResNet18 model (without pretrained ImageNet weights)
        self.model = models.resnet18(pretrained=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
        # If using pre-trained weights from Imagenette, load them
        if use_pretrained_imagenette and imagenette_checkpoint_path is not None:
            # Load checkpoint (adjust loading as needed for your checkpoint structure)
            state_dict = torch.load(imagenette_checkpoint_path, map_location=self.device)
            # Load state dict into the model (using strict=False to ignore mismatched keys if any)
            self.model.load_state_dict(state_dict, strict=False)
        
        self.learning_rate = learning_rate
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = F.cross_entropy(logits, labels)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

if __name__ == '__main__':
    # Set parameters
    data_dir = "/Users/tylbuchanan/Documents/Python Projects/CSE 6363 - Machine Learning/Assignment 1 - Linear Models/CSE6363/assignments/deep_learning/cifar-10-batches-py"
    batch_size = 128
    num_workers = 4
    # Set this flag to True to fine-tune the model using pre-trained Imagenette weights
    use_pretrained_imagenette = True  
    # Path to the checkpoint file from your Imagenette training run
    imagenette_checkpoint_path = "/Users/tylbuchanan/Documents/Python Projects/CSE 6363 - Machine Learning/Assignment 1 - Linear Models/CSE6363/assignments/deep_learning/lightning_logs/version_1/checkpoints/epoch=13-step=3318.ckpt"  

    # Prepare CIFAR10 DataModule
    cifar10_dm = CIFAR10DataModule(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)
    cifar10_dm.prepare_data()
    cifar10_dm.setup()

    # Initialize the model; switch use_pretrained_imagenette flag as needed for your experiments
    model_transfer = ResNet18TransferModule(num_classes=10, learning_rate=1e-3,
                                              use_pretrained_imagenette=use_pretrained_imagenette,
                                              imagenette_checkpoint_path=imagenette_checkpoint_path)
    
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    if torch.cuda.is_available():
        trainer = pl.Trainer(max_epochs=50, callbacks=[early_stop_callback], accelerator="gpu", devices=1)
    elif torch.backends.mps.is_available():
        trainer = pl.Trainer(max_epochs=50, callbacks=[early_stop_callback], accelerator="mps", devices=1)
    else:
        trainer = pl.Trainer(max_epochs=50, callbacks=[early_stop_callback], accelerator="cpu")

    
    # Train and then validate the model on CIFAR10
    trainer.fit(model_transfer, cifar10_dm)
    trainer.validate(model_transfer, cifar10_dm)
