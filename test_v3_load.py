
import sys
import os

# Add the backend directory to sys.path so we can import 'app'
sys.path.append(os.getcwd())

import torch
try:
    from app.services.inference_v3 import get_inference_engine
    print("Import successful.")
    
    engine = get_inference_engine()
    print("Engine initialized.")
    print(f"Device: {engine.device}")
    print(f"FP16: {engine.half_precision}")
    
    print("Test passed.")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
