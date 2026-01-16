#!/usr/bin/env python
"""Script de prueba para verificar que el motor de IA funciona correctamente"""

import sys
import os

# Configurar encoding UTF-8
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')  # type: ignore

# Agregar el path del backend
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Probar que todos los imports funcionan"""
    print("🧪 Probando imports...")
    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
        
        import torchvision
        print(f"  ✓ torchvision {torchvision.__version__}")
        
        from PIL import Image
        print(f"  ✓ PIL")
        
        from app.services.inference import (
            get_inference_engine,
            PhytoClassifier,
            InferenceEngine
        )
        print(f"  ✓ app.services.inference")
        
        print("\n✅ Todos los imports funcionan correctamente\n")
        return True
    except Exception as e:
        print(f"\n❌ Error en imports: {e}\n")
        return False


def test_inference_engine():
    """Probar que el motor de inferencia se inicializa"""
    print("🧪 Probando motor de inferencia...")
    try:
        from app.services.inference import get_inference_engine
        
        engine = get_inference_engine()
        print(f"  ✓ Motor de inferencia inicializado")
        
        info = engine.get_model_info()
        print(f"  ✓ Dispositivo: {info['device']}")
        print(f"  ✓ Modelo: {info['model_name']}")
        print(f"  ✓ Clases: {', '.join(info['classes'])}")
        print(f"  ✓ Tamaño de imagen: {info['image_size']}")
        
        print("\n✅ Motor de inferencia funciona correctamente\n")
        return True
    except Exception as e:
        print(f"\n❌ Error en motor de inferencia: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_endpoints():
    """Probar que los endpoints se importan correctamente"""
    print("🧪 Probando endpoints...")
    try:
        from app.api.v1.endpoints.scans import router
        print(f"  ✓ Endpoints de scans importados")
        print(f"  ✓ Número de rutas: {len(router.routes)}")
        
        for route in router.routes:
            if hasattr(route, 'path'):
                print(f"    - {route.methods if hasattr(route, 'methods') else 'GET'} {route.path}")
        
        print("\n✅ Endpoints funcionan correctamente\n")
        return True
    except Exception as e:
        print(f"\n❌ Error en endpoints: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_schemas():
    """Probar que los esquemas se importan correctamente"""
    print("🧪 Probando esquemas...")
    try:
        from app.schemas.scan import (
            Scan,
            ScanResult,
            ScanResponse,
            ScanCreate
        )
        print(f"  ✓ Esquemas de escaneo importados")
        
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
        print(f"  ✓ ScanResult creado: {result.label} ({result.confidence:.0%})")
        
        print("\n✅ Esquemas funcionan correctamente\n")
        return True
    except Exception as e:
        print(f"\n❌ Error en esquemas: {e}\n")
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
        status = "✅ PASADO" if result else "❌ FALLIDO"
        print(f"{name:.<40} {status}")
    
    all_passed = all(result for _, result in results)
    
    print("=" * 60)
    if all_passed:
        print("🎉 ¡TODAS LAS PRUEBAS PASARON! El sistema está listo.")
    else:
        print("⚠️  Algunas pruebas fallaron. Revisa los errores arriba.")
    print("=" * 60)
    
    sys.exit(0 if all_passed else 1)
