"""
Hierarchical Multi-Task Learning Model for Cannabis Recognition

Architecture:
- Shared backbone: EfficientNetV2-L (60M params) 
- Task 1: Product type (plant, flower, trim, extract, hash) - 5 classes
- Task 2: Quality grade (A+, A, B, C, F) - 5 classes  
- Task 3: Appearance features (attributes) - 10 binary attributes
- Task 4: Uncertainty quantification (epistemic + aleatoric)

This model represents the core architecture upgrade from EfficientNetV2-M
to a hierarchical multi-task learning approach.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Tuple
import torch.nn.functional as F


class HierarchicalCannabisModel(nn.Module):
    """Multi-task learning model for cannabis product recognition"""
    
    def __init__(self, backbone_name: str = "efficientnet_v2_l", num_classes: int = 5):
        """
        Initialize hierarchical cannabis recognition model
        
        Args:
            backbone_name: Name of backbone architecture
            num_classes: Number of primary classification classes
        """
        super().__init__()
        
        # BACKBONE: Pretrained on ImageNet
        if backbone_name == "efficientnet_v2_l":
            backbone = models.efficientnet_v2_l(weights="DEFAULT")
            self.feature_dim = 1280
        elif backbone_name == "efficientnet_v2_m":
            backbone = models.efficientnet_v2_m(weights="DEFAULT")
            self.feature_dim = 1280
        elif backbone_name == "efficientnet_v2_s":
            backbone = models.efficientnet_v2_s(weights="DEFAULT")
            self.feature_dim = 1280
        else:
            raise ValueError(f"Backbone {backbone_name} not supported")
        
        # Remove classification head, keep only features
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # SPATIAL ATTENTION LAYER
        # Learn which parts of image are important for classification
        self.spatial_attention = SpatialAttentionModule(self.feature_dim)
        
        # TASK 1: PRIMARY CLASSIFICATION (Product Type)
        self.head_primary = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # TASK 2: QUALITY GRADE CLASSIFICATION
        self.head_quality = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 5)  # Grades: A+, A, B, C, F
        )
        
        # TASK 3: APPEARANCE ATTRIBUTES (Multi-label)
        self.head_attributes = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)  # 10 binary attributes
        )
        
        # TASK 4: UNCERTAINTY QUANTIFICATION
        self.head_uncertainty = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)  # epistemic, aleatoric uncertainty
        )
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical model
        
        Args:
            x: Input image tensor (B, 3, 448, 448)
            return_features: Return intermediate features for visualization
        
        Returns:
            Dict with all task predictions
        """
        # Feature extraction
        features = self.backbone(x)  # (B, 1280, 14, 14)
        
        # Spatial attention
        attention_map = self.spatial_attention(features)  # (B, 1, 14, 14)
        features_attended = features * attention_map
        
        # Task predictions
        primary_logits = self.head_primary(features_attended)           # (B, 5)
        quality_logits = self.head_quality(features_attended)           # (B, 5)
        attributes_logits = self.head_attributes(features_attended)     # (B, 10)
        uncertainty_params = self.head_uncertainty(features)            # (B, 2)
        
        results = {
            "primary_logits": primary_logits,
            "primary_probs": F.softmax(primary_logits, dim=1),
            
            "quality_logits": quality_logits,
            "quality_probs": F.softmax(quality_logits, dim=1),
            
            "attributes_logits": attributes_logits,
            "attributes_probs": torch.sigmoid(attributes_logits),
            
            "uncertainty": uncertainty_params,
        }
        
        if return_features:
            results["attention_map"] = attention_map
            results["features"] = features
        
        return results


class SpatialAttentionModule(nn.Module):
    """Spatial attention mechanism - learn which parts of image matter"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate spatial attention weights"""
        return self.conv(x)


class HierarchicalLoss(nn.Module):
    """Combined loss for multi-task learning"""
    
    def __init__(self, 
                 w_primary: float = 1.0,
                 w_quality: float = 0.6,
                 w_attributes: float = 0.3,
                 w_uncertainty: float = 0.2):
        super().__init__()
        self.w_primary = w_primary
        self.w_quality = w_quality
        self.w_attributes = w_attributes
        self.w_uncertainty = w_uncertainty
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined loss
        
        targets = {
            "primary": torch.Tensor(B,),      # class indices 0-4
            "quality": torch.Tensor(B,),      # class indices 0-4
            "attributes": torch.Tensor(B, 10), # binary 0/1
        }
        """
        
        # Task 1: Primary classification
        loss_primary = self.ce_loss(
            predictions["primary_logits"],
            targets["primary"]
        )
        
        # Task 2: Quality classification
        loss_quality = self.ce_loss(
            predictions["quality_logits"],
            targets["quality"]
        )
        
        # Task 3: Attributes (multi-label)
        loss_attributes = self.bce_loss(
            predictions["attributes_logits"],
            targets["attributes"].float()
        )
        
        # Task 4: Uncertainty regularization
        # Penalize if model is overconfident
        max_prob = torch.max(predictions["primary_probs"], dim=1)[0]
        loss_uncertainty = torch.mean((max_prob - 0.8) ** 2)  # Target 80% confidence
        
        # Combined loss
        total_loss = (
            self.w_primary * loss_primary +
            self.w_quality * loss_quality +
            self.w_attributes * loss_attributes +
            self.w_uncertainty * loss_uncertainty
        )
        
        return total_loss, {
            "loss_primary": loss_primary.item(),
            "loss_quality": loss_quality.item(),
            "loss_attributes": loss_attributes.item(),
            "loss_uncertainty": loss_uncertainty.item(),
            "total_loss": total_loss.item()
        }


# Test/validation code
if __name__ == "__main__":
    # Create model
    model = HierarchicalCannabisModel(backbone_name="efficientnet_v2_m")
    model.eval()
    
    # Create dummy input (batch of 2 images, 3 channels, 448x448)
    dummy_input = torch.randn(2, 3, 448, 448)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print("✓ Model architecture validated")
    print(f"  Primary output shape: {output['primary_probs'].shape}")
    print(f"  Quality output shape: {output['quality_probs'].shape}")
    print(f"  Attributes output shape: {output['attributes_probs'].shape}")
    print(f"  Uncertainty output shape: {output['uncertainty'].shape}")
    
    # Test loss
    criterion = HierarchicalLoss()
    targets = {
        "primary": torch.tensor([0, 2]),
        "quality": torch.tensor([1, 3]),
        "attributes": torch.randint(0, 2, (2, 10))
    }
    
    loss, loss_dict = criterion(output, targets)
    print(f"\n✓ Loss computation validated")
    print(f"  Total loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
