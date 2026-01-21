#!/usr/bin/env python
"""
DATASET AUDIT SCRIPT - Week 1 Task 1.2

Audita el dataset actual e identifica gaps.
Genera DATASET_AUDIT.json con desglose completo.
"""

import os
import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Common paths where datasets might be stored
POSSIBLE_DATA_PATHS = [
    "data",
    "dataset",
    "datasets",
    "../data",
    "../../data",
    "data/cannabis",
    "data/images",
    "/data",
    "C:/Users/ilyas/datasets",
    "C:/data",
    os.path.expanduser("~/data"),
    os.path.expanduser("~/datasets"),
]

# Cannabis product classes
CLASSES = {
    "plant": "Live cannabis plants",
    "flower": "Dried flower buds",
    "trim": "Trim and small buds",
    "hash": "Hash and concentrates",
    "extract": "Oil extracts and distillates"
}

# Quality grades
GRADES = ["A+", "A", "B", "C", "F"]

# Appearance attributes
ATTRIBUTES = [
    "dense_structure",
    "color_vibrancy",
    "trichome_coverage",
    "moisture_level",
    "mold_presence",
    "leaf_ratio",
    "seed_presence",
    "stem_ratio",
    "crystal_coverage",
    "overall_condition"
]

def find_image_files(directory):
    """Find all image files in directory"""
    if not os.path.exists(directory):
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    images = []
    
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    images.append(os.path.join(root, file))
    except PermissionError:
        pass
    
    return images

def locate_dataset():
    """Find the dataset directory"""
    print("=" * 80)
    print("LOCATING DATASET")
    print("=" * 80)
    
    for path in POSSIBLE_DATA_PATHS:
        if os.path.exists(path):
            print(f"[FOUND] {path}")
            return path
    
    print("[INFO] Dataset directory not found in standard locations")
    print("[INFO] Creating sample dataset structure for reference...")
    return None

def parse_filename(filename):
    """Parse filename to extract metadata"""
    # Expected format: class_grade_device_metadata.jpg
    # Example: flower_A_iphone_natural_light.jpg
    
    basename = Path(filename).stem
    parts = basename.split('_')
    
    metadata = {
        "filename": Path(filename).name,
        "class": None,
        "grade": None,
        "device": None,
        "lighting": None,
    }
    
    if len(parts) >= 1 and parts[0] in CLASSES:
        metadata["class"] = parts[0]
    
    if len(parts) >= 2 and parts[1] in GRADES:
        metadata["grade"] = parts[1]
    
    if len(parts) >= 3:
        metadata["device"] = parts[2]
    
    return metadata

def audit_dataset(dataset_path):
    """Perform complete dataset audit"""
    print("\n" + "=" * 80)
    print("AUDITING DATASET")
    print("=" * 80)
    
    images = find_image_files(dataset_path) if dataset_path else []
    
    print(f"Total images found: {len(images)}")
    
    audit = {
        "timestamp": datetime.now().isoformat(),
        "dataset_path": dataset_path,
        "total_images": len(images),
        "by_class": defaultdict(lambda: {"count": 0, "grades": defaultdict(int), "devices": defaultdict(int)}),
        "by_grade": defaultdict(int),
        "by_device": defaultdict(int),
        "unclassified": [],
        "statistics": {}
    }
    
    # Parse all images
    for img_path in images:
        metadata = parse_filename(img_path)
        
        if metadata["class"]:
            cls = metadata["class"]
            audit["by_class"][cls]["count"] += 1
            
            if metadata["grade"]:
                audit["by_class"][cls]["grades"][metadata["grade"]] += 1
                audit["by_grade"][metadata["grade"]] += 1
            
            if metadata["device"]:
                audit["by_class"][cls]["devices"][metadata["device"]] += 1
                audit["by_device"][metadata["device"]] += 1
        else:
            audit["unclassified"].append(Path(img_path).name)
    
    return audit

def identify_gaps(audit):
    """Identify gaps in dataset coverage"""
    print("\n" + "=" * 80)
    print("IDENTIFYING GAPS")
    print("=" * 80)
    
    total = audit["total_images"]
    gaps = {
        "classes": {},
        "grades": {},
        "device_coverage": {},
        "recommendations": []
    }
    
    # Check class distribution
    print("\nClass Distribution:")
    for cls in CLASSES.keys():
        count = audit["by_class"][cls]["count"]
        percentage = (count / total * 100) if total > 0 else 0
        status = "OK" if percentage >= 15 else "GAP"
        print(f"  {cls:12} {count:5} images ({percentage:5.1f}%) [{status}]")
        gaps["classes"][cls] = {
            "count": count,
            "percentage": percentage,
            "needs_data": percentage < 15
        }
    
    # Check grade distribution
    print("\nGrade Distribution:")
    for grade in GRADES:
        count = audit["by_grade"][grade]
        percentage = (count / total * 100) if total > 0 else 0
        status = "OK" if percentage >= 10 else "GAP"
        print(f"  Grade {grade:2} {count:5} images ({percentage:5.1f}%) [{status}]")
        gaps["grades"][grade] = {
            "count": count,
            "percentage": percentage,
            "needs_data": percentage < 10
        }
    
    # Device coverage
    print("\nDevice Coverage:")
    for device, count in sorted(audit["by_device"].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {device:15} {count:5} images ({percentage:5.1f}%)")
        gaps["device_coverage"][device] = {
            "count": count,
            "percentage": percentage
        }
    
    # Generate recommendations
    gaps["recommendations"] = generate_recommendations(audit, gaps)
    
    return gaps

def generate_recommendations(audit, gaps):
    """Generate data collection recommendations"""
    total = audit["total_images"]
    recommendations = []
    
    # Class recommendations
    for cls, data in gaps["classes"].items():
        if data["needs_data"]:
            target = int(total * 0.20)  # Target 20% per class
            needed = max(0, target - data["count"])
            if needed > 0:
                recommendations.append({
                    "type": "class",
                    "category": cls,
                    "current": data["count"],
                    "target": target,
                    "needed": needed,
                    "priority": "HIGH" if data["percentage"] < 5 else "MEDIUM"
                })
    
    # Grade recommendations
    for grade, data in gaps["grades"].items():
        if data["needs_data"]:
            target = int(total * 0.15)  # Target 15% per grade
            needed = max(0, target - data["count"])
            if needed > 0:
                recommendations.append({
                    "type": "grade",
                    "category": grade,
                    "current": data["count"],
                    "target": target,
                    "needed": needed,
                    "priority": "MEDIUM"
                })
    
    # Device coverage recommendations
    device_types = ["iPhone", "Android", "Webcam", "Professional"]
    for device in device_types:
        count = gaps["device_coverage"].get(device, {}).get("count", 0)
        if count == 0:
            recommendations.append({
                "type": "device",
                "category": device,
                "current": count,
                "target": int(total * 0.25),
                "needed": int(total * 0.25),
                "priority": "MEDIUM"
            })
    
    return recommendations

def save_audit_report(audit, gaps):
    """Save audit results to JSON files"""
    print("\n" + "=" * 80)
    print("SAVING REPORTS")
    print("=" * 80)
    
    # Convert defaultdicts to regular dicts for JSON serialization
    audit_clean = {
        "timestamp": audit["timestamp"],
        "dataset_path": audit["dataset_path"],
        "total_images": audit["total_images"],
        "by_class": {k: {
            "count": v["count"],
            "grades": dict(v["grades"]),
            "devices": dict(v["devices"])
        } for k, v in audit["by_class"].items()},
        "by_grade": dict(audit["by_grade"]),
        "by_device": dict(audit["by_device"]),
        "unclassified_count": len(audit["unclassified"]),
        "unclassified_samples": audit["unclassified"][:10]  # First 10
    }
    
    gaps_clean = {
        "classes": gaps["classes"],
        "grades": gaps["grades"],
        "device_coverage": gaps["device_coverage"],
        "recommendations": gaps["recommendations"]
    }
    
    # Save audit
    with open("DATASET_AUDIT.json", "w") as f:
        json.dump(audit_clean, f, indent=2)
    print("[SAVED] DATASET_AUDIT.json")
    
    # Save gaps
    with open("DATASET_GAPS.json", "w") as f:
        json.dump(gaps_clean, f, indent=2)
    print("[SAVED] DATASET_GAPS.json")

def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + "DATASET AUDIT - Week 1 Task 1.2".center(78) + "║")
    print("║" + "Comprehensive inventory and gap analysis".center(78) + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Locate dataset
    dataset_path = locate_dataset()
    
    # Perform audit
    audit = audit_dataset(dataset_path)
    
    # Identify gaps
    gaps = identify_gaps(audit)
    
    # Save reports
    save_audit_report(audit, gaps)
    
    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print(f"Total images audited: {audit['total_images']}")
    print(f"Files generated:")
    print(f"  - DATASET_AUDIT.json (detailed inventory)")
    print(f"  - DATASET_GAPS.json (gap analysis & recommendations)")
    print(f"\nNext steps:")
    print(f"  1. Review the audit reports")
    print(f"  2. Create DATA_COLLECTION_PLAN.md")
    print(f"  3. Continue with Task 1.3 (Monitoring Setup)")

if __name__ == "__main__":
    main()
