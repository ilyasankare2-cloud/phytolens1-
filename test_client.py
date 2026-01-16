#!/usr/bin/env python
"""
Cliente de prueba para la API de PhytoLens
Prueba los endpoints disponibles
"""

import sys
import os
import time
import requests
import io
from PIL import Image, ImageDraw

# URL del servidor
API_URL = "http://127.0.0.1:8001"

def create_test_image():
    """Crear imagen de prueba"""
    img = Image.new('RGB', (224, 224), color=(73, 109, 137))
    draw = ImageDraw.Draw(img)
    
    draw.ellipse([50, 50, 150, 150], fill=(34, 139, 34), outline=(0, 100, 0))
    draw.ellipse([100, 80, 160, 140], fill=(50, 150, 50), outline=(0, 100, 0))
    draw.rectangle([105, 150, 120, 190], fill=(139, 69, 19), outline=(101, 50, 10))
    
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes


def test_model_info():
    """Probar endpoint de info del modelo"""
    print("[TEST] Probando GET /model-info...")
    
    try:
        response = requests.get(f"{API_URL}/api/v1/scans/model-info")
        
        if response.status_code == 200:
            data = response.json()
            print("  OK: Status 200")
            print(f"    - Modelo: {data.get('model_name')}")
            print(f"    - Dispositivo: {data.get('device')}")
            print(f"    - Clases: {', '.join(data.get('classes', []))}")
            return True
        else:
            print(f"  ERROR: Status {response.status_code}")
            print(f"    - {response.text}")
            return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_root():
    """Probar endpoint raiz"""
    print("[TEST] Probando GET /...")
    
    try:
        response = requests.get(f"{API_URL}/")
        
        if response.status_code == 200:
            print("  OK: Status 200")
            print(f"    - {response.json()}")
            return True
        else:
            print(f"  ERROR: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_docs():
    """Probar documentacion swagger"""
    print("[TEST] Probando GET /docs (Swagger UI)...")
    
    try:
        response = requests.get(f"{API_URL}/docs")
        
        if response.status_code == 200:
            print("  OK: Status 200")
            print("    - Documentacion Swagger disponible")
            return True
        else:
            print(f"  ERROR: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    """Ejecutar pruebas del cliente"""
    
    print("=" * 70)
    print("PRUEBA DEL CLIENTE - PhytoLens API")
    print("=" * 70)
    print()
    
    # Esperar a que el servidor inicie
    print("[INFO] Esperando a que el servidor inicie...")
    time.sleep(2)
    
    print("[INFO] Conectando a: " + API_URL)
    print()
    
    results = []
    
    # Probar endpoints
    results.append(("Root endpoint", test_root()))
    print()
    
    results.append(("Model info", test_model_info()))
    print()
    
    results.append(("Swagger UI", test_docs()))
    print()
    
    # Resumen
    print("=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print()
    
    for name, result in results:
        status = "OK" if result else "ERROR"
        print(f"  {name:.<40} [{status}]")
    
    print()
    print("=" * 70)
    print("INFORMACION UTIL")
    print("=" * 70)
    print()
    print(f"API URL:               {API_URL}")
    print(f"Swagger UI:            {API_URL}/docs")
    print(f"ReDoc:                 {API_URL}/redoc")
    print(f"OpenAPI Schema:        {API_URL}/api/v1/openapi.json")
    print()
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("RESULTADO: SERVIDOR OPERATIVO!")
    else:
        print("RESULTADO: ALGUNOS ENDPOINTS FALLARON")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
