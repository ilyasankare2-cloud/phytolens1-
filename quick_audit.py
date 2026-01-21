#!/usr/bin/env python
"""
Fast dataset audit with explicit output
"""
import os
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

CLASSES = ["plant", "flower", "trim", "hash", "extract"]
GRADES = ["A+", "A", "B", "C", "F"]

def find_images():
    """Find all images"""
    paths = ["data", "dataset", "datasets", "../data", "../../data"]
    image_ext = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = []
    
    for path in paths:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if Path(file).suffix.lower() in image_ext:
                        images.append(os.path.join(root, file))
    
    return images

print("\n" + "="*80)
print("DATASET AUDIT - Week 1 Task 1.2")
print("="*80 + "\n")

images = find_images()
print(f"[*] Total images found: {len(images)}")

if len(images) == 0:
    print("[!] No images found - creating sample audit structure\n")
    
    # Create default/sample audit for reference
    sample_audit = {
        "timestamp": datetime.now().isoformat(),
        "total_images": 0,
        "status": "NO_DATASET_FOUND",
        "by_class": {cls: {"count": 0, "grades": {g: 0 for g in GRADES}} for cls in CLASSES},
        "by_grade": {g: 0 for g in GRADES},
        "notes": "Dataset not found. Please populate data directory and re-run audit.",
        "next_steps": [
            "1. Create data/ directory",
            "2. Add cannabis plant images",
            "3. Name files as: class_grade_device_metadata.jpg",
            "4. Example: flower_A_iphone_indoor.jpg",
            "5. Re-run audit_dataset.py"
        ]
    }
else:
    print(f"[*] Parsing {len(images)} images...\n")
    
    audit = {
        "timestamp": datetime.now().isoformat(),
        "total_images": len(images),
        "by_class": defaultdict(lambda: {"count": 0, "grades": defaultdict(int)}),
        "by_grade": defaultdict(int),
    }
    
    for img in images:
        basename = Path(img).stem.split('_')[0].lower()
        if basename in CLASSES:
            cls = basename
            audit["by_class"][cls]["count"] += 1
    
    sample_audit = {
        "timestamp": audit["timestamp"],
        "total_images": audit["total_images"],
        "by_class": {k: dict(v) for k, v in audit["by_class"].items()},
        "by_grade": dict(audit["by_grade"])
    }
    
    print("Class Distribution:")
    for cls in CLASSES:
        count = sample_audit["by_class"].get(cls, {}).get("count", 0)
        pct = (count / len(images) * 100) if len(images) > 0 else 0
        print(f"  {cls:10} {count:5} images ({pct:5.1f}%)")

# Save audit
with open("DATASET_AUDIT.json", "w") as f:
    json.dump(sample_audit, f, indent=2)

print(f"\n[OK] DATASET_AUDIT.json created")
print(f"[OK] Audit complete\n")
