#!/usr/bin/env python
"""
Servidor FastAPI de demostración para PhytoLens IA
Permite probar la IA sin requerir base de datos
"""

import sys
import os
import io
from datetime import datetime
from typing import List

# Agregar path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from app.services.inference import get_inference_engine

# Crear app
app = FastAPI(
    title="PhytoLens IA - Demo",
    description="API de demostración para clasificación de plantas con IA",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Esquemas
class PredictionResult(BaseModel):
    label: str
    confidence: float
    all_probabilities: dict

class AnalysisResponse(BaseModel):
    success: bool
    timestamp: str
    result: PredictionResult
    message: str = ""

class ModelInfo(BaseModel):
    device: str
    model_name: str
    num_classes: int
    classes: List[str]
    image_size: int


# Endpoints
@app.get("/", tags=["Health"])
def root():
    """Endpoint raiz"""
    return {
        "message": "PhytoLens IA - Demo API",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "analyze": "/analyze",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health", tags=["Health"])
def health():
    """Verificar salud del servidor"""
    return {
        "status": "healthy",
        "service": "PhytoLens IA",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model-info", response_model=ModelInfo, tags=["Model"])
def get_model_info():
    """Obtener información del modelo de IA"""
    engine = get_inference_engine()
    info = engine.get_model_info()
    return ModelInfo(**info)


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_image(file: UploadFile = File(...)):
    """
    Analizar una imagen de planta y obtener clasificación.
    
    - **file**: Archivo de imagen (JPEG, PNG, WebP)
    - **Retorna**: Clasificación, confianza y probabilidades
    """
    try:
        # Validar tipo de archivo
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tipo de archivo no soportado. Usar: {', '.join(allowed_extensions)}"
            )
        
        # Leer archivo
        contents = await file.read()
        
        # Validar que sea imagen
        try:
            img = Image.open(io.BytesIO(contents))
            img.verify()
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Archivo no es una imagen válida"
            )
        
        # Realizar predicción
        engine = get_inference_engine()
        result = engine.predict(contents)
        
        return AnalysisResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            result=PredictionResult(**result),
            message=f"Imagen analizada correctamente. Clase detectada: {result['label']}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar imagen: {str(e)}"
        )


@app.post("/analyze-batch", tags=["Analysis"])
async def analyze_batch(files: List[UploadFile] = File(...)):
    """
    Analizar múltiples imágenes.
    
    - **files**: Lista de archivos de imagen
    - **Retorna**: Lista de resultados de predicción
    """
    results = []
    errors = []
    
    engine = get_inference_engine()
    
    for idx, file in enumerate(files):
        try:
            contents = await file.read()
            result = engine.predict(contents)
            
            results.append({
                "file": file.filename,
                "success": True,
                "result": result
            })
        except Exception as e:
            errors.append({
                "file": file.filename,
                "error": str(e)
            })
    
    return {
        "total": len(files),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("SERVIDOR FASTAPI - PhytoLens IA Demo")
    print("=" * 70)
    print()
    print("Iniciando servidor...")
    print()
    print("URL:              http://127.0.0.1:8001")
    print("Swagger UI:       http://127.0.0.1:8001/docs")
    print("ReDoc:            http://127.0.0.1:8001/redoc")
    print()
    print("=" * 70)
    print()
    
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
