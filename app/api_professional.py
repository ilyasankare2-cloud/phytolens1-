"""
API profesional mejorada para VisionPlant
Incluye validación exhaustiva, manejo de errores, optimizaciones y seguridad
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, status, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZIPMiddleware
import logging
import os
from typing import Optional
from datetime import datetime
import aiofiles
import asyncio
from contextlib import asynccontextmanager

# Configuración de logging profesional
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar motor de IA
from app.services.inference import get_inference_engine

# Constantes
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff'}
UPLOAD_DIR = "uploads"
CLEANUP_INTERVAL = 3600  # 1 hora

# Crear directorio de uploads
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Variables globales
inference_engine = None
cleanup_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestor del ciclo de vida de la aplicación"""
    global inference_engine, cleanup_task
    
    # Startup
    logger.info("🚀 Iniciando VisionPlant...")
    try:
        inference_engine = get_inference_engine(use_tta=False)
        logger.info("✓ Motor de IA cargado correctamente")
    except Exception as e:
        logger.error(f"✗ Error cargando motor de IA: {e}")
        raise
    
    # Iniciar limpieza de archivos viejos
    async def cleanup_old_files():
        while True:
            try:
                await asyncio.sleep(CLEANUP_INTERVAL)
                current_time = datetime.now().timestamp()
                for filename in os.listdir(UPLOAD_DIR):
                    filepath = os.path.join(UPLOAD_DIR, filename)
                    if os.path.isfile(filepath):
                        if current_time - os.path.getmtime(filepath) > 86400:  # 24 horas
                            os.remove(filepath)
                            logger.debug(f"Archivo limpiado: {filename}")
            except Exception as e:
                logger.error(f"Error limpiando archivos: {e}")
    
    cleanup_task = asyncio.create_task(cleanup_old_files())
    
    yield
    
    # Shutdown
    logger.info("🛑 Deteniendo VisionPlant...")
    if cleanup_task:
        cleanup_task.cancel()
    logger.info("✓ VisionPlant detenido correctamente")

# Crear aplicación
app = FastAPI(
    title="VisionPlant API",
    description="API profesional de reconocimiento de plantas con IA avanzada",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(GZIPMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# ============================================================================
# MODELOS DE RESPUESTA
# ============================================================================

from pydantic import BaseModel

class PredictionResult(BaseModel):
    label: str
    confidence: float
    certainty: float
    model_version: str
    top_3_predictions: list
    all_probabilities: dict

class AnalysisResponse(BaseModel):
    success: bool
    result: PredictionResult
    processing_time_ms: float
    timestamp: str

class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    image_size: int
    num_classes: int
    tta_enabled: bool
    device: str
    supported_formats: list

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

async def validate_upload_file(file: UploadFile) -> tuple[bool, Optional[str]]:
    """Validar archivo subido"""
    # Validar nombre
    if not file.filename:
        return False, "Nombre de archivo inválido"
    
    # Validar extensión
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False, f"Formato no soportado. Usar: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Validar tamaño (lectura de primeros bytes)
    content = await file.read(MAX_FILE_SIZE + 1)
    await file.seek(0)
    
    if len(content) > MAX_FILE_SIZE:
        return False, f"Archivo demasiado grande (máximo {MAX_FILE_SIZE // (1024*1024)}MB)"
    
    if len(content) == 0:
        return False, "Archivo vacío"
    
    return True, None

async def save_upload_file(file: UploadFile, user_id: str = "anonymous") -> str:
    """Guardar archivo subido de forma segura"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{user_id}_{timestamp}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    try:
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(await file.read())
        logger.info(f"Archivo guardado: {filename}")
        return filepath
    except Exception as e:
        logger.error(f"Error guardando archivo: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        raise

async def cleanup_file(filepath: str, background_tasks: BackgroundTasks):
    """Programar eliminación de archivo después de cierto tiempo"""
    async def delete_file():
        await asyncio.sleep(3600)  # 1 hora
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.debug(f"Archivo eliminado automáticamente: {filepath}")
        except Exception as e:
            logger.warning(f"Error eliminando archivo {filepath}: {e}")
    
    background_tasks.add_task(delete_file)

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Web"])
async def root():
    """Servir página principal"""
    return FileResponse("app/templates/index.html", media_type="text/html")

@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health_check():
    """Verificar estado del sistema"""
    return {
        "status": "healthy",
        "model_loaded": inference_engine is not None,
        "version": "1.0.0"
    }

@app.get("/api/v1/model-info", response_model=ModelInfo, tags=["Información"])
async def get_model_info():
    """Obtener información del modelo"""
    if not inference_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Motor de IA no disponible"
        )
    
    try:
        info = inference_engine.get_model_info()
        return {
            "model_name": info.get("model_name", "EfficientNetV2-M"),
            "model_version": info.get("model_version", "V2 Improved"),
            "image_size": info.get("image_size", 384),
            "num_classes": info.get("num_classes", 5),
            "tta_enabled": info.get("tta_enabled", False),
            "device": info.get("device", "cpu"),
            "supported_formats": list(ALLOWED_EXTENSIONS)
        }
    except Exception as e:
        logger.error(f"Error obteniendo información del modelo: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error obteniendo información del modelo"
        )

@app.post("/api/v1/analyze", response_model=AnalysisResponse, tags=["Análisis"])
async def analyze_image(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Analizar imagen de planta con IA avanzada
    
    - Soporta: JPG, PNG, GIF, WebP, BMP, TIFF
    - Máximo: 10MB
    - Retorna: Clasificación, confianza y probabilidades
    """
    
    if not inference_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Motor de IA no disponible"
        )
    
    start_time = datetime.now()
    filepath = None
    
    try:
        # 1. Validar archivo
        is_valid, error_msg = await validate_upload_file(file)
        if not is_valid:
            logger.warning(f"Validación fallida: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        
        # 2. Guardar archivo
        filepath = await save_upload_file(file)
        
        # 3. Leer archivo para predicción
        async with aiofiles.open(filepath, 'rb') as f:
            image_bytes = await f.read()
        
        # 4. Realizar predicción
        logger.info(f"Analizando imagen: {file.filename}")
        result = inference_engine.predict(image_bytes)
        
        # 5. Calcular tiempo de procesamiento
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # 6. Programar eliminación de archivo
        await cleanup_file(filepath, background_tasks)
        
        # 7. Loguear resultado
        logger.info(
            f"Análisis completado: {result['label']} "
            f"(confianza: {result['confidence']:.2%}) "
            f"en {processing_time:.0f}ms"
        )
        
        return {
            "success": True,
            "result": result,
            "processing_time_ms": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en análisis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error procesando la imagen. Intenta con otra imagen."
        )
    finally:
        # Limpiar archivo temporal si hubo error
        if filepath and os.path.exists(filepath):
            try:
                # No eliminar inmediatamente en caso de error para debugging
                pass
            except Exception as e:
                logger.warning(f"Error en cleanup: {e}")

@app.post("/api/v1/analyze-batch", tags=["Análisis"])
async def analyze_batch(files: list[UploadFile] = File(...)):
    """
    Analizar múltiples imágenes
    - Máximo: 10 imágenes
    - Procesa todas secuencialmente
    """
    
    if not inference_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Motor de IA no disponible"
        )
    
    if len(files) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Máximo 10 imágenes por solicitud"
        )
    
    results = []
    
    for file in files:
        try:
            # Validar
            is_valid, error_msg = await validate_upload_file(file)
            if not is_valid:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": error_msg
                })
                continue
            
            # Guardar
            filepath = await save_upload_file(file)
            
            # Leer y analizar
            async with aiofiles.open(filepath, 'rb') as f:
                image_bytes = await f.read()
            
            result = inference_engine.predict(image_bytes)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "result": result
            })
            
            # Limpiar
            if os.path.exists(filepath):
                os.remove(filepath)
                
        except Exception as e:
            logger.error(f"Error en batch {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {"results": results}

# ============================================================================
# MANEJO DE ERRORES GLOBAL
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Manejar excepciones HTTP"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Manejar excepciones generales"""
    logger.error(f"Error no manejado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Error interno del servidor",
            "timestamp": datetime.now().isoformat()
        }
    )

# ============================================================================
# DOCUMENTACIÓN Y METADATOS
# ============================================================================

app.openapi_tags = [
    {
        "name": "Web",
        "description": "Endpoints de interfaz web"
    },
    {
        "name": "Sistema",
        "description": "Endpoints de sistema y salud"
    },
    {
        "name": "Información",
        "description": "Información del modelo"
    },
    {
        "name": "Análisis",
        "description": "Endpoints de análisis de imágenes"
    }
]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_professional:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=4,
        log_level="info"
    )
