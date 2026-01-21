"""
🎓 Fine-tuning y aprendizaje personalizado
Entrenar modelo con datos específicos del usuario
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

logger = logging.getLogger(__name__)


class CustomPlantDataset(Dataset):
    """Dataset personalizado para fine-tuning"""
    
    def __init__(
        self,
        image_dir: str,
        annotations_file: str,
        transforms=None
    ):
        """
        Args:
            image_dir: Directorio con imágenes
            annotations_file: JSON con {filename: class_name}
            transforms: Transformaciones de imagen
        """
        self.image_dir = Path(image_dir)
        self.transforms = transforms
        
        with open(annotations_file) as f:
            self.annotations = json.load(f)
        
        self.class_names = ['plant', 'dry_flower', 'resin', 'extract', 'processed']
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.samples = list(self.annotations.items())
        
        logger.info(f"✓ Dataset: {len(self.samples)} muestras")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename, class_name = self.samples[idx]
        image_path = self.image_dir / filename
        
        # Cargar imagen
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        
        # Aplicar transformaciones
        if self.transforms:
            image = self.transforms(image)
        
        label = self.class_to_idx[class_name]
        
        return image, label


class FineTuningEngine:
    """Motor de fine-tuning"""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = 'cuda'
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=1e-6
        )
        
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, float]:
        """Entrenar una época"""
        self.model.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Métricas
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / total_samples
        
        # Validación
        val_loss = None
        val_accuracy = None
        if val_loader:
            val_loss, val_accuracy = self._validate(val_loader)
        
        # Scheduler
        self.scheduler.step()
        
        # Guardar historial
        self.history['train_loss'].append(train_loss)
        self.history['train_accuracy'].append(train_accuracy)
        if val_loss:
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
        
        return {
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validar modelo"""
        self.model.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        return total_loss / len(val_loader), total_correct / total_samples
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        save_path: Optional[str] = None
    ) -> Dict:
        """Entrenar modelo"""
        logger.info(f"🎓 Iniciando fine-tuning por {epochs} épocas")
        
        best_val_accuracy = 0
        
        for epoch in range(epochs):
            metrics = self.train_epoch(train_loader, val_loader)
            
            log_msg = (
                f"Época {epoch+1}/{epochs} | "
                f"Train Loss: {metrics['train_loss']:.4f} | "
                f"Train Acc: {metrics['train_accuracy']:.4f}"
            )
            
            if metrics['val_loss'] is not None:
                log_msg += (
                    f" | Val Loss: {metrics['val_loss']:.4f} | "
                    f"Val Acc: {metrics['val_accuracy']:.4f}"
                )
                
                # Guardar mejor modelo
                if metrics['val_accuracy'] > best_val_accuracy:
                    best_val_accuracy = metrics['val_accuracy']
                    if save_path:
                        self.save_checkpoint(save_path, epoch)
            
            logger.info(log_msg)
        
        logger.info(f"✓ Fine-tuning completado. Best val acc: {best_val_accuracy:.4f}")
        
        return {
            'best_val_accuracy': best_val_accuracy,
            'history': self.history
        }
    
    def save_checkpoint(self, path: str, epoch: int):
        """Guardar checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history
        }
        
        torch.save(checkpoint, path)
        logger.info(f"✓ Checkpoint guardado: {path}")
    
    def load_checkpoint(self, path: str):
        """Cargar checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.history = checkpoint['history']
        logger.info(f"✓ Checkpoint cargado: {path}")
    
    def plot_history(self, save_path: Optional[str] = None):
        """Graficar historial de entrenamiento"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        if self.history['val_loss']:
            axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(self.history['train_accuracy'], label='Train Acc')
        if self.history['val_accuracy']:
            axes[1].plot(self.history['val_accuracy'], label='Val Acc')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"✓ Gráfico guardado: {save_path}")
        
        return fig


class FineTuningPipeline:
    """Pipeline completo de fine-tuning"""
    
    def __init__(
        self,
        model: nn.Module,
        dataset_dir: str,
        output_dir: str = "finetuned_models",
        device: str = 'cuda'
    ):
        self.model = model
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = device
        
        self.engine = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def setup_training_data(
        self,
        train_split: float = 0.8,
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader]:
        """Preparar data loaders"""
        from torchvision import transforms
        
        # Transformaciones
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Dataset
        annotations_file = self.dataset_dir / "annotations.json"
        dataset = CustomPlantDataset(
            str(self.dataset_dir / "images"),
            str(annotations_file),
            transforms=train_transforms
        )
        
        # Split
        train_size = int(len(dataset) * train_split)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, val_loader
    
    def run(
        self,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-4
    ) -> Dict:
        """Ejecutar pipeline completo"""
        logger.info("=" * 60)
        logger.info(f"🎓 Iniciando pipeline de fine-tuning")
        logger.info(f"Dataset: {self.dataset_dir}")
        logger.info("=" * 60)
        
        # Preparar data
        train_loader, val_loader = self.setup_training_data(
            batch_size=batch_size
        )
        
        # Motor de fine-tuning
        self.engine = FineTuningEngine(
            self.model,
            learning_rate=learning_rate,
            device=self.device
        )
        
        # Entrenar
        model_path = self.output_dir / f"finetuned_{self.timestamp}.pt"
        results = self.engine.train(
            train_loader,
            val_loader,
            epochs=epochs,
            save_path=str(model_path)
        )
        
        # Graficar
        plot_path = self.output_dir / f"training_{self.timestamp}.png"
        self.engine.plot_history(str(plot_path))
        
        # Guardar metadata
        metadata = {
            'timestamp': self.timestamp,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'best_val_accuracy': results['best_val_accuracy'],
            'history': results['history']
        }
        
        metadata_path = self.output_dir / f"metadata_{self.timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("=" * 60)
        logger.info(f"✓ Fine-tuning completado")
        logger.info(f"Modelo: {model_path}")
        logger.info(f"Metadata: {metadata_path}")
        logger.info("=" * 60)
        
        return {
            'model_path': str(model_path),
            'metadata_path': str(metadata_path),
            'results': results
        }
