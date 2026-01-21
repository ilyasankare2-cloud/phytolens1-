#!/usr/bin/env python
"""
Quick training test (2 batches only, 1 epoch)
"""
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
from app.models import HierarchicalCannabisModel
from app.models.hierarchical_model import HierarchicalLoss
import torch.optim as optim

print("Creating quick test dataset...")
X = torch.randn(8, 3, 224, 224)
y_primary = torch.randint(0, 5, (8,))
y_quality = torch.randint(0, 5, (8,))
y_attributes = torch.randint(0, 2, (8, 10)).float()

dataset = TensorDataset(X, y_primary, y_quality, y_attributes)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

print("Initializing model...")
model = HierarchicalCannabisModel(backbone_name="efficientnet_v2_s")
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = HierarchicalLoss(w_primary=1.0, w_quality=0.6, w_attributes=0.3, w_uncertainty=0.2)

print("Training for 1 epoch (2 batches)...")
model.train()
total_loss = 0

for batch_idx, (images, prim, qual, attr) in enumerate(loader):
    if batch_idx >= 2:  # Only 2 batches
        break
    
    print(f"  Batch {batch_idx + 1}...")
    
    optimizer.zero_grad()
    
    outputs = model(images)
    
    targets = {
        "primary": prim,
        "quality": qual,
        "attributes": attr
    }
    
    loss, loss_dict = criterion(outputs, targets)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    total_loss += loss.item()
    print(f"    Loss: {loss.item():.4f}")

print()
print("=" * 60)
print("[OK] Training completed successfully!")
print("=" * 60)
print()
print("Next:")
print("  1. Save checkpoint")
print("  2. Verify file exists")
print("  3. Continue with full training or next task")
