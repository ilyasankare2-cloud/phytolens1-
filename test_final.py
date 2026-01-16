#!/usr/bin/env python
"""Script de prueba para verificar que el motor de IA funciona correctamente"""

import sys
import os

# Agregar el path del backend
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Probar que todos los imports funcionan"""
    print("[TEST] Probando imports...")
    try:
        import torch
        print(f"  OK: torch {torch.__version__}")
        
        import torchvision
        print(f"  OK: torchvision {torchvision.__version__}")
        
        from PIL import Image
        print(f"  OK: PIL")
        
        from app.services.inference import (
            get_inference_engine,
            PhytoClassifier,
            InferenceEngine
        )
        print(f"  OK: app.services.inference")
        
        print("\nPASO 1: IMPORTS - EXITO\n")
        return True
    except Exception as e:
        print(f"\nPASO 1: IMPORTS - ERROR: {e}\n")
        return False


def test_inference_engine():
    """Probar que el motor de inferencia se inicializa"""
    print("[TEST] Probando motor de inferencia...")
    try:
        from app.services.inference import get_inference_engine
        
        engine = get_inference_engine()
        print(f"  OK: Motor de inferencia inicializado")
        
        info = engine.get_model_info()
        print(f"  OK: Dispositivo: {info['device']}")
        print(f"  OK: Modelo: {info['model_name']}")
        print(f"  OK: Clases: {', '.join(info['classes'])}")
        print(f"  OK: Tamanho de imagen: {info['image_size']}")
        
        print("\nPASO 2: MOTOR DE IA - EXITO\n")
        return True
    except Exception as e:
        print(f"\nPASO 2: MOTOR DE IA - ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_endpoints():
    """Probar que los endpoints se importan correctamente"""
    print("[TEST] Probando endpoints...")
    try:
        from app.api.v1.endpoints.scans import router
        print(f"  OK: Endpoints de scans importados")
        print(f"  OK: Numero de rutas: {len(router.routes)}")
        
        for route in router.routes:
            if hasattr(route, 'path'):
                methods = str(route.methods) if hasattr(route, 'methods') else 'GET'
                print(f"    - {methods} {route.path}")
        
        print("\nPASO 3: ENDPOINTS - EXITO\n")
        return True
    except Exception as e:
        print(f"\nPASO 3: ENDPOINTS - ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_schemas():
    """Probar que los esquemas se importan correctamente"""
    print("[TEST] Probando esquemas...")
    try:
        from app.schemas.scan import (
            Scan,
            ScanResult,
            ScanResponse,
            ScanCreate
        )
        print(f"  OK: Esquemas de escaneo importados")
        
        # Crear instancia de ejemplo
        result = ScanResult(
            label="plant",
            confidence=0.95,
            all_probabilities={
                "plant": 0.95,
                "dry_flower": 0.03,
                "resin": 0.01,
                "extract": 0.005,
                "processed": 0.005
            }
        )
        print(f"  OK: ScanResult creado: {result.label} ({result.confidence:.0%})")
        
        print("\nPASO 4: ESQUEMAS - EXITO\n")
        return True
    except Exception as e:
        print(f"\nPASO 4: ESQUEMAS - ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("PRUEBA DE SISTEMA - PhytoLens IA Backend")
    print("=" * 60 + "\n")
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Esquemas", test_schemas()))
    results.append(("Motor de IA", test_inference_engine()))
    results.append(("Endpoints", test_endpoints()))
    
    print("=" * 60)
    print("RESUMEN DE PRUEBAS")
    print("=" * 60)
    
    for name, result in results:
        status = "OK" if result else "ERROR"
        print(f"{name:.<40} [{status}]")
    
    all_passed = all(result for _, result in results)
    
    print("=" * 60)
    if all_passed:
        print("RESULTADO: TODAS LAS PRUEBAS PASARON!")
        print("Sistema listo para produccion")
    else:
        print("RESULTADO: ALGUNAS PRUEBAS FALLARON")
        print("Revisa los errores arriba")
    print("=" * 60)
    
    sys.exit(0 if all_passed else 1)
