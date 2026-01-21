#!/usr/bin/env python
import sys
import torch

print("=" * 80)
print("TEST 1: PyTorch")
print("=" * 80)
print(f"[OK] PyTorch: {torch.__version__}")
print(f"[OK] GPU: {torch.cuda.is_available()}")
print()

print("=" * 80)
print("TEST 2: Import HierarchicalCannabisModel")
print("=" * 80)
try:
    from app.models import HierarchicalCannabisModel
    print("[OK] Importacion exitosa")
except Exception as e:
    print(f"[FAIL] Error: {e}")
    sys.exit(1)
print()

print("=" * 80)
print("TEST 3: Create model instance")
print("=" * 80)
try:
    model = HierarchicalCannabisModel(backbone_name="efficientnet_v2_s")
    params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Modelo creado: HierarchicalCannabisModel")
    print(f"[OK] Parametros: {params:,}")
except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 80)
print("TEST 4: Forward pass (batch_size=2)")
print("=" * 80)
try:
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print("[OK] Forward pass completado")
    print(f"[OK] Outputs: {list(output.keys())}")
    for key, value in output.items():
        print(f"     - {key}: {value.shape}")
except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 80)
print("TEST 5: Loss function")
print("=" * 80)
try:
    from app.models.hierarchical_model import HierarchicalLoss
    
    model.train()
    
    criterion = HierarchicalLoss(
        w_primary=1.0,
        w_quality=0.6,
        w_attributes=0.3,
        w_uncertainty=0.2
    )
    print("[OK] HierarchicalLoss creado")
    
    target_primary = torch.randint(0, 5, (2,))
    target_quality = torch.randint(0, 5, (2,))
    target_attributes = torch.randint(0, 2, (2, 10)).float()
    
    targets = {
        "primary": target_primary,
        "quality": target_quality,
        "attributes": target_attributes
    }
    
    loss, loss_dict = criterion(output, targets)
    
    print("[OK] Loss computation completada")
    print(f"     Total loss: {loss.item():.4f}")
    print("     Loss breakdown:")
    for key, val in loss_dict.items():
        print(f"       - {key}: {val:.4f}")
except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print()
print("Next steps:")
print("  1. Run: python scripts/train_hierarchical.py --epochs 2 --batch-size 4")
print("  2. Check: checkpoints/best_model.pt")
print("  3. Continue with Week 1 Task 1.2 (Dataset Audit)")

