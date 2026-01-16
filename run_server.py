#!/usr/bin/env python
"""
Script para desplegar el servidor FastAPI de PhytoLens sin BD
Para desarrollo y pruebas locales
"""

import sys
import os

# Agregar el path del backend
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Desplegar el servidor FastAPI"""
    
    print("=" * 70)
    print("DESPLIEGUE DEL SERVIDOR - PhytoLens IA Backend")
    print("=" * 70)
    print()
    
    # Verificar uvicorn
    try:
        import uvicorn
    except ImportError:
        print("[ERROR] uvicorn no instalado")
        print("Instalando uvicorn...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "uvicorn"])
        import uvicorn
    
    print("[INFO] Iniciando servidor FastAPI...")
    print()
    print("=" * 70)
    print("SERVIDOR INICIADO")
    print("=" * 70)
    print()
    print("URL de la API:         http://localhost:8000")
    print("Documentacion Swagger: http://localhost:8000/docs")
    print("Documentacion ReDoc:   http://localhost:8000/redoc")
    print()
    print("Endpoints disponibles:")
    print("  POST   /api/v1/scans/analyze    - Analizar imagen")
    print("  GET    /api/v1/scans/model-info - Info del modelo IA")
    print("  GET    /api/v1/scans            - Listar escaneos")
    print("  GET    /api/v1/scans/{id}       - Obtener escaneo")
    print("  DELETE /api/v1/scans/{id}       - Eliminar escaneo")
    print()
    print("Presiona Ctrl+C para detener")
    print("=" * 70)
    print()
    
    # Ejecutar servidor
    try:
        uvicorn.run(
            "app.main:app",
            host="127.0.0.1",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("Servidor detenido")
        print("=" * 70)


if __name__ == "__main__":
    main()
