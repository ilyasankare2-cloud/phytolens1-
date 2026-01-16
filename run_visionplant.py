#!/usr/bin/env python
"""
VisionPlant - Servidor profesional
App profesional de reconocimiento de plantas con IA avanzada
"""

import sys
import os
import logging

# Agregar ruta del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visionplant.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def print_banner():
    """Mostrar banner de inicio"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🌿 VISIONPLANT - Reconocimiento de Plantas con IA 🌿  ║
    ║                                                           ║
    ║              Version 1.0 - PRODUCCIÓN                    ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Punto de entrada principal"""
    print_banner()
    
    logger.info("=" * 60)
    logger.info("VisionPlant - Iniciando servidor profesional")
    logger.info("=" * 60)
    
    try:
        # Crear directorio de templates si no existe
        os.makedirs("app/templates", exist_ok=True)
        os.makedirs("app/static", exist_ok=True)
        os.makedirs("uploads", exist_ok=True)
        
        logger.info("✓ Directorios inicializados")
        
        # Importar y ejecutar app
        import uvicorn
        from app.api_professional import app
        
        logger.info("✓ Aplicación importada correctamente")
        logger.info("")
        logger.info("🚀 Iniciando servidor en http://0.0.0.0:8000")
        logger.info("📊 Documentación: http://localhost:8000/docs")
        logger.info("🎨 Interfaz: http://localhost:8000")
        logger.info("")
        
        # Ejecutar servidor con configuración optimizada
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            workers=4,
            loop="uvloop",
            log_level="info",
            access_log=True
        )
        
    except ImportError as e:
        logger.error(f"✗ Error importando dependencias: {e}")
        logger.error("Instala las dependencias necesarias:")
        logger.error("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"✗ Error fatal: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
