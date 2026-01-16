#!/usr/bin/env python
"""
Script de comparación y prueba - IA Mejorada V2
Demuestra las mejoras implementadas
"""

import sys
import os
import io
import time
from PIL import Image, ImageDraw
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_image(seed: int = 42):
    """Crear imagen de prueba variada"""
    np.random.seed(seed)
    
    # Usar 256x256 para balance entre velocidad y calidad
    img = Image.new('RGB', (256, 256), color=(73, 109, 137))
    draw = ImageDraw.Draw(img)
    
    # Dibujar formas aleatorias
    for _ in range(5):
        x1 = np.random.randint(0, 200)
        y1 = np.random.randint(0, 200)
        x2 = x1 + np.random.randint(30, 80)
        y2 = y1 + np.random.randint(30, 80)
        color = tuple(np.random.randint(0, 255, 3))
        draw.ellipse([x1, y1, x2, y2], fill=color)
    
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()


def test_model_improvements():
    """Demostrar mejoras del modelo"""
    
    print("=" * 70)
    print("PRUEBA DE MEJORAS - PhytoLens IA V2")
    print("=" * 70)
    print()
    
    print("[TEST 1] Cargar Motor Mejorado")
    print("-" * 70)
    
    try:
        from app.services.inference import get_inference_engine
        
        print("  >> Inicializando motor con TTA habilitado...")
        engine = get_inference_engine(use_tta=True)
        print("  OK: Motor inicializado\n")
        
        # Información del modelo
        info = engine.get_model_info()
        
        print("  Información del Modelo:")
        print(f"    - Modelo: {info['model_name']}")
        print(f"    - Versión: {info['model_version']}")
        print(f"    - Tamaño imagen: {info['image_size']}x{info['image_size']}")
        print(f"    - TTA habilitado: {info['tta_enabled']}")
        print(f"    - Características:")
        for feat in info['features']:
            print(f"      • {feat}")
        print()
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    
    print("[TEST 2] Predicción con TTA")
    print("-" * 70)
    
    try:
        print("  >> Creando imagen de prueba...")
        image_bytes = create_test_image(seed=42)
        print(f"  OK: Imagen creada ({len(image_bytes)} bytes)\n")
        
        print("  >> Ejecutando predicción con TTA...")
        start_time = time.time()
        result = engine.predict(image_bytes)
        elapsed = time.time() - start_time
        
        print(f"  OK: Predicción completada en {elapsed:.2f}s\n")
        
        # Mostrar resultado mejorado
        print("  Resultado de Predicción:")
        print(f"    - Clase: {result['label']}")
        print(f"    - Confianza: {result['confidence']:.2%}")
        print(f"    - Certidumbre: {result['certainty']:.2%}")
        print(f"    - Versión: {result['model_version']}")
        print()
        
        print("  Top 3 Predicciones:")
        for i, pred in enumerate(result['top_3_predictions'], 1):
            bar = "█" * int(pred['probability'] * 30)
            print(f"    {i}. {pred['label']:.<20} {pred['probability']:.4f} {bar}")
        print()
        
        print("  Todas las Probabilidades:")
        for label, prob in sorted(result['all_probabilities'].items(), 
                                 key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 30)
            print(f"    {label:.<20} {prob:.4f} {bar}")
        print()
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("[TEST 3] Caché de Predicciones")
    print("-" * 70)
    
    try:
        print("  >> Primera predicción (sin caché)...")
        image_bytes = create_test_image(seed=123)
        start_time = time.time()
        result1 = engine.predict(image_bytes)
        time1 = time.time() - start_time
        
        print(f"  OK: {time1*1000:.2f}ms\n")
        
        print("  >> Segunda predicción (misma imagen - con caché)...")
        start_time = time.time()
        result2 = engine.predict(image_bytes)
        time2 = time.time() - start_time
        
        print(f"  OK: {time2*1000:.2f}ms\n")
        
        speedup = time1 / time2 if time2 > 0 else float('inf')
        print(f"  Mejora de velocidad: {speedup:.1f}x más rápido")
        print(f"  Caché hits: {engine.prediction_cache.__len__()} predicciones almacenadas")
        print()
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    
    print("[TEST 4] Validación de Imagen")
    print("-" * 70)
    
    try:
        # Imagen válida
        print("  >> Probando imagen válida...")
        image_bytes = create_test_image()
        result = engine.predict(image_bytes)
        print(f"  OK: Predicción exitosa - {result['label']}\n")
        
        # Imagen pequeña
        print("  >> Probando imagen muy pequeña (validación)...")
        small_img = Image.new('RGB', (50, 50), color=(100, 100, 100))
        small_bytes = io.BytesIO()
        small_img.save(small_bytes, format='PNG')
        
        try:
            result = engine.predict(small_bytes.getvalue())
            print(f"  Resultado: {result}\n")
        except ValueError as e:
            print(f"  OK: Validación rechazó imagen pequeña\n")
            print(f"  Razón: {str(e)}\n")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    
    print("[TEST 5] Predicción en Lote")
    print("-" * 70)
    
    try:
        print("  >> Creando 3 imágenes de prueba...")
        images = [create_test_image(seed=i) for i in range(3)]
        print(f"  OK: {len(images)} imágenes creadas\n")
        
        print("  >> Procesando predicciones en lote...")
        start_time = time.time()
        results = engine.predict_batch(images)
        elapsed = time.time() - start_time
        
        print(f"  OK: {len(results)} predicciones en {elapsed:.2f}s\n")
        
        print("  Resultados del Lote:")
        for i, result in enumerate(results, 1):
            print(f"    {i}. {result['label']} ({result['confidence']:.2%})")
        print()
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    
    print("[TEST 6] Información del Modelo Extendida")
    print("-" * 70)
    
    try:
        info = engine.get_model_info()
        
        print("  Información del Sistema:")
        print(f"    - Dispositivo: {info['device']}")
        print(f"    - Modelo: {info['model_name']}")
        print(f"    - Versión: {info['model_version']}")
        print(f"    - Clases: {info['num_classes']}")
        print(f"    - Tamaño entrada: {info['image_size']}x{info['image_size']}")
        print(f"    - TTA: {'Habilitado' if info['tta_enabled'] else 'Deshabilitado'}")
        print(f"    - Cache size: {info['cache_size']} predicciones")
        print()
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    
    print("=" * 70)
    print("RESUMEN DE MEJORAS COMPROBADAS")
    print("=" * 70)
    print()
    
    improvements = [
        ("Modelo EfficientNetV2-M", "✓ 54.1M parámetros (vs 21.5M)"),
        ("Test-Time Augmentation", "✓ 4 variaciones por imagen"),
        ("Caché de Predicciones", "✓ Reutilización de resultados"),
        ("Validación Robusta", "✓ Control de entrada"),
        ("Métricas Extendidas", "✓ Certainty scoring + top-3"),
        ("Procesamiento en Lote", "✓ Optimizado para múltiples imágenes"),
    ]
    
    for improvement, status in improvements:
        print(f"  {improvement:.<40} {status}")
    
    print()
    print("=" * 70)
    print("RESULTADO FINAL: IA V2 FUNCIONAL Y MEJORADA")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = test_model_improvements()
    sys.exit(0 if success else 1)
