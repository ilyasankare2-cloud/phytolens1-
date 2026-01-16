#!/usr/bin/env python
"""
Script Simple - IA Mejorada V2
Demostración rápida sin TTA
"""

import sys
import os
import io
import time
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_image(seed: int = 42):
    """Crear imagen de prueba"""
    np.random.seed(seed)
    img = Image.new('RGB', (224, 224), color=(73, 109, 137))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.getvalue()


def main():
    print("\n" + "="*60)
    print("PhytoLens IA V2 - DEMOSTRACIÓN RÁPIDA")
    print("="*60 + "\n")
    
    try:
        # Importar motor
        print("[1] Inicializando motor...")
        from app.services.inference import get_inference_engine
        
        # Sin TTA para mayor velocidad
        engine = get_inference_engine(use_tta=False)
        print("✓ Motor inicializado\n")
        
        # Información
        print("[2] Información del modelo:")
        info = engine.get_model_info()
        print(f"   - Modelo: {info['model_name']}")
        print(f"   - Versión: {info['model_version']}")
        print(f"   - Tamaño: {info['image_size']}x{info['image_size']}")
        print()
        
        # Predicción simple
        print("[3] Realizando predicción...")
        image = create_test_image()
        start = time.time()
        result = engine.predict(image)
        elapsed = time.time() - start
        
        print(f"✓ Predicción en {elapsed:.2f}s\n")
        
        print("[4] Resultado:")
        print(f"   - Clase: {result['label']}")
        print(f"   - Confianza: {result['confidence']:.2%}")
        print(f"   - Certidumbre: {result['certainty']:.2%}")
        print()
        
        print("[5] Distribución de Probabilidades:")
        for label, prob in sorted(result['all_probabilities'].items(), 
                                 key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 40)
            print(f"   {label:.<20} {prob:.4f} {bar}")
        print()
        
        print("[6] Top 3 Predicciones:")
        for i, pred in enumerate(result['top_3_predictions'], 1):
            print(f"   {i}. {pred['label']} ({pred['probability']:.2%})")
        print()
        
        print("="*60)
        print("✓ IA V2 FUNCIONAL Y LISTA PARA USAR")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
