#!/usr/bin/env python
"""Script para probar la IA con una imagen de prueba"""

import sys
import os
import io
from PIL import Image, ImageDraw
import numpy as np

# Agregar el path del backend
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_image():
    """Crear una imagen de prueba"""
    print("[TEST] Creando imagen de prueba...")
    
    # Crear imagen RGB de 224x224
    img = Image.new('RGB', (224, 224), color=(73, 109, 137))
    draw = ImageDraw.Draw(img)
    
    # Dibujar formas para simular una planta
    draw.ellipse([50, 50, 150, 150], fill=(34, 139, 34), outline=(0, 100, 0))
    draw.ellipse([100, 80, 160, 140], fill=(50, 150, 50), outline=(0, 100, 0))
    draw.rectangle([105, 150, 120, 190], fill=(139, 69, 19), outline=(101, 50, 10))
    
    # Guardar a bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    print("  OK: Imagen creada (224x224 PNG)")
    return img_bytes.getvalue()


def test_inference():
    """Probar la IA con una imagen"""
    print("\n[TEST] Probando motor de IA...\n")
    
    try:
        from app.services.inference import get_inference_engine
        
        # Obtener motor de IA
        print("  >> Inicializando motor de IA...")
        engine = get_inference_engine()
        print("  OK: Motor inicializado\n")
        
        # Crear imagen de prueba
        image_bytes = create_test_image()
        print()
        
        # Realizar predicción
        print("  >> Ejecutando predicción...")
        result = engine.predict(image_bytes)
        print("  OK: Predicción completada\n")
        
        # Mostrar resultados
        print("=" * 60)
        print("RESULTADO DE LA PREDICCION")
        print("=" * 60)
        print(f"\nClase detectada: {result['label']}")
        print(f"Confianza: {result['confidence']:.2%}")
        print(f"\nProbabilidades por clase:")
        
        for label, prob in sorted(result['all_probabilities'].items(), 
                                 key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 30)
            print(f"  {label:.<20} {prob:.4f} {bar}")
        
        print("\n" + "=" * 60)
        
        # Información del modelo
        print("\nINFORMACION DEL MODELO")
        print("=" * 60)
        info = engine.get_model_info()
        print(f"Dispositivo: {info['device']}")
        print(f"Modelo: {info['model_name']}")
        print(f"Clases: {', '.join(info['classes'])}")
        print(f"Tamanho entrada: {info['image_size']}x{info['image_size']}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_prediction():
    """Probar predicción en lote"""
    print("\n\n[TEST] Probando prediccion en lote...\n")
    
    try:
        from app.services.inference import get_inference_engine
        
        engine = get_inference_engine()
        
        # Crear 3 imágenes de prueba
        print("  >> Creando 3 imágenes de prueba...")
        images = [create_test_image() for _ in range(3)]
        print(f"  OK: {len(images)} imágenes creadas\n")
        
        # Predicción en lote
        print("  >> Ejecutando predicciones en lote...")
        results = engine.predict_batch(images)
        print(f"  OK: {len(results)} predicciones completadas\n")
        
        print("=" * 60)
        print("RESULTADOS DEL LOTE")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"\nImagen {i}: {result['label']} ({result['confidence']:.2%})")
        
        print("\n" + "=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("PRUEBA DE EJECUCION - PhytoLens IA")
    print("=" * 60 + "\n")
    
    results = []
    
    # Prueba 1: Predicción simple
    results.append(("Prediccion simple", test_inference()))
    
    # Prueba 2: Predicción en lote
    results.append(("Prediccion en lote", test_batch_prediction()))
    
    # Resumen
    print("\n\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    
    for name, result in results:
        status = "OK" if result else "ERROR"
        print(f"{name:.<40} [{status}]")
    
    all_passed = all(result for _, result in results)
    
    print("=" * 60)
    if all_passed:
        print("\nRESULTADO: LA IA FUNCIONA CORRECTAMENTE!")
        print("Motor de IA listo para produccion")
    else:
        print("\nRESULTADO: ALGUNAS PRUEBAS FALLARON")
    print("=" * 60)
    
    sys.exit(0 if all_passed else 1)
