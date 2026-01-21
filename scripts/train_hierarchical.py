"""
Training script for hierarchical cannabis recognition model

Usage:
    python scripts/train_hierarchical.py --epochs 20 --lr 1e-4 --batch-size 32
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
from pathlib import Path
from datetime import datetime
import logging
import argparse
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.hierarchical_model import HierarchicalCannabisModel, HierarchicalLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DummyPlantDataset(Dataset):
    """Dummy dataset for testing - replace with real data"""
    
    def __init__(self, num_samples: int = 100, img_size: int = 448):
        self.num_samples = num_samples
        self.img_size = img_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Dummy image
        image = torch.randn(3, self.img_size, self.img_size)
        
        # Dummy labels
        primary_label = torch.randint(0, 5, (1,)).item()
        quality_label = torch.randint(0, 5, (1,)).item()
        attributes = torch.randint(0, 2, (10,))
        
        return {
            "image": image,
            "primary": primary_label,
            "quality": quality_label,
            "attributes": attributes
        }


class HierarchicalTrainer:
    """Trainer for hierarchical cannabis recognition model"""
    
    def __init__(self, 
                 model: HierarchicalCannabisModel,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 learning_rate: float = 1e-4,
                 checkpoint_dir: str = "checkpoints"):
        
        self.model = model.to(device)
        self.device = device
        self.criterion = HierarchicalLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_primary_acc": [], "val_primary_acc": [],
            "train_quality_acc": [], "val_quality_acc": [],
        }
        
        logger.info(f"Training on device: {device}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(self.device)
            
            targets = {
                "primary": torch.tensor(batch["primary"]).to(self.device),
                "quality": torch.tensor(batch["quality"]).to(self.device),
                "attributes": batch["attributes"].to(self.device)
            }
            
            # Forward pass
            predictions = self.model(images)
            
            # Loss
            loss, loss_breakdown = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}: Loss={loss.item():.4f}")
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> dict:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct_primary = 0
        correct_quality = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                targets = {
                    "primary": torch.tensor(batch["primary"]).to(self.device),
                    "quality": torch.tensor(batch["quality"]).to(self.device),
                    "attributes": batch["attributes"].to(self.device)
                }
                
                predictions = self.model(images)
                loss, _ = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                
                # Accuracy
                primary_pred = predictions["primary_probs"].argmax(dim=1)
                quality_pred = predictions["quality_probs"].argmax(dim=1)
                
                correct_primary += (primary_pred == targets["primary"]).sum().item()
                correct_quality += (quality_pred == targets["quality"]).sum().item()
                total_samples += images.shape[0]
        
        return {
            "val_loss": total_loss / len(val_loader),
            "primary_accuracy": correct_primary / total_samples if total_samples > 0 else 0,
            "quality_accuracy": correct_quality / total_samples if total_samples > 0 else 0,
        }
    
    def fit(self, 
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 20):
        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_primary_acc"].append(val_metrics["primary_accuracy"])
            self.history["val_quality_acc"].append(val_metrics["quality_accuracy"])
            
            self.scheduler.step()
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Primary Acc: {val_metrics['primary_accuracy']:.4f} | "
                f"Quality Acc: {val_metrics['quality_accuracy']:.4f}"
            )
            
            # Save best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                torch.save(self.model.state_dict(), 
                          self.checkpoint_dir / "best_model.pt")
                logger.info("✓ New best model saved")
        
        # Save history
        with open(self.checkpoint_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        
        logger.info("✓ Training complete")


def main():
    parser = argparse.ArgumentParser(description='Train hierarchical cannabis model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("HIERARCHICAL CANNABIS MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Device: {args.device}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Create model
    model = HierarchicalCannabisModel(backbone_name="efficientnet_v2_m")
    logger.info("✓ Model created")
    
    # Create dummy datasets (replace with real data)
    train_dataset = DummyPlantDataset(num_samples=100)
    val_dataset = DummyPlantDataset(num_samples=20)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    logger.info(f"✓ Datasets created (train={len(train_dataset)}, val={len(val_dataset)})")
    
    # Create trainer
    trainer = HierarchicalTrainer(
        model=model,
        device=args.device,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train
    logger.info("Starting training...")
    trainer.fit(train_loader, val_loader, epochs=args.epochs)
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best model saved to: {trainer.checkpoint_dir}/best_model.pt")
    logger.info(f"History saved to: {trainer.checkpoint_dir}/history.json")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
