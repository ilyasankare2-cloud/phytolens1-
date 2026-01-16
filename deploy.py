#!/usr/bin/env python
"""
Script para desplegar el servidor FastAPI de PhytoLens
Ejecuta la API en puerto 8000 con auto-reload
"""

import sys
import os
import subprocess

# Agregar el path del backend
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def deploy():
    """Desplegar el servidor FastAPI"""
    
    print("=" * 70)
    print("DESPLIEGUE DEL SERVIDOR - PhytoLens IA Backend")
    print("=" * 70)
    print()
    
    # Verificar que uvicorn está instalado
    print("[INFO] Verificando dependencias...")
    try:
        import uvicorn
        print("  OK: uvicorn instalado")
    except ImportError:
        print("  ERROR: uvicorn no está instalado")
        print("  Instalando uvicorn...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "uvicorn"])
    
    print()
    print("[INFO] Iniciando servidor FastAPI...")
    print()
    print("=" * 70)
    print("SERVIDOR INICIADO")
    print("=" * 70)
    print()
    print("URL de la API:        http://localhost:8000")
    print("Documentacion Swagger: http://localhost:8000/docs")
    print("Documentacion ReDoc:  http://localhost:8000/redoc")
    print()
    print("Endpoints disponibles:")
    print("  POST   /api/v1/scans/analyze    - Analizar imagen")
    print("  GET    /api/v1/scans/model-info - Info del modelo")
    print("  GET    /api/v1/scans            - Listar escaneos")
    print("  GET    /api/v1/scans/{id}       - Obtener escaneo")
    print("  DELETE /api/v1/scans/{id}       - Eliminar escaneo")
    print()
    print("Presiona Ctrl+C para detener el servidor")
    print()
    print("=" * 70)
    print()
    
    # Ejecutar el servidor
    try:
        import uvicorn
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=["app"],
            log_level="info"
        )
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("Servidor detenido")
        print("=" * 70)
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    deploy()
