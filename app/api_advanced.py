"""
🚀 API Extended - Nuevos endpoints para características avanzadas
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
import io
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/advanced", tags=["Advanced Features"])


# ============================================================================
# MODELOS PYDANTIC
# ============================================================================

class CacheStats(BaseModel):
    """Estadísticas de caché"""
    hit_rate: str
    total_hits: int
    total_misses: int
    memory_items: int
    disk_items: int
    disk_size_mb: float
    redis_enabled: bool


class ExplanationResponse(BaseModel):
    """Respuesta con explicación"""
    label: str
    confidence: float
    heatmap_base64: Optional[str]
    confidence_analysis: Dict[str, Any]
    top_3: List[Dict[str, Any]]


class EdgeModelInfo(BaseModel):
    """Información de modelo edge"""
    name: str
    architecture: str
    parameters: Dict[str, Any]
    input_size: int
    num_classes: int
    quantized: bool
    device: str
    estimated_size_mb: float


class EdgePredictionResponse(BaseModel):
    """Respuesta de modelo edge"""
    label: str
    confidence: float
    inference_time_ms: float
    device: str


class FineTuningConfig(BaseModel):
    """Configuración de fine-tuning"""
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    train_split: float = 0.8


# ============================================================================
# ENDPOINTS: CACHÉ
# ============================================================================

cache_manager = None  # Se inicializa desde main


@router.get("/cache/stats", response_model=CacheStats)
async def get_cache_stats():
    """Obtener estadísticas del caché"""
    if not cache_manager:
        raise HTTPException(status_code=503, detail="Cache no disponible")
    
    return cache_manager.get_stats()


@router.post("/cache/clear")
async def clear_cache(expired_only: bool = True):
    """Limpiar caché"""
    if not cache_manager:
        raise HTTPException(status_code=503, detail="Cache no disponible")
    
    if expired_only:
        cache_manager.clear_expired()
        return {"status": "ok", "message": "Entradas expiradas eliminadas"}
    else:
        cache_manager.clear_all()
        return {"status": "ok", "message": "Caché completamente limpio"}


# ============================================================================
# ENDPOINTS: EXPLICABILIDAD
# ============================================================================

explainability_engine = None  # Se inicializa desde main


@router.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(file: UploadFile = File(...)):
    """
    Analizar imagen con explicabilidad (GRAD-CAM)
    
    Retorna:
    - Heatmap visual de activación
    - Análisis de confianza adaptativo
    - Top-3 predicciones
    """
    if not explainability_engine:
        raise HTTPException(status_code=503, detail="Motor de explicabilidad no disponible")
    
    try:
        image_bytes = await file.read()
        
        # Cargar imagen original
        original_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        original_array = np.array(original_image)
        
        # Predicción
        from app.services.inference_optimized import get_inference_engine
        engine = get_inference_engine()
        result = engine.predict(image_bytes)
        
        # Obtener tensor para GRAD-CAM
        image_tensor = engine._prepare_image(image_bytes)
        predicted_class_idx = ['plant', 'dry_flower', 'resin', 'extract', 'processed'].index(result['label'])
        
        # Generar explicación
        explanation = explainability_engine.explain_prediction(
            image_tensor,
            np.array([result['all_probabilities'][c] for c in ['plant', 'dry_flower', 'resin', 'extract', 'processed']]),
            predicted_class_idx,
            original_array,
            include_heatmap=True
        )
        
        return ExplanationResponse(
            label=result['label'],
            confidence=result['confidence'],
            heatmap_base64=explanation.get('heatmap_base64'),
            confidence_analysis=explanation['confidence_analysis'],
            top_3=explanation['top_3']
        )
    
    except Exception as e:
        logger.error(f"Error en explicabilidad: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# ENDPOINTS: MODELO EDGE
# ============================================================================

edge_model = None  # Se inicializa desde main


@router.get("/edge/info", response_model=EdgeModelInfo)
async def get_edge_model_info():
    """Información del modelo edge"""
    if not edge_model:
        raise HTTPException(status_code=503, detail="Modelo edge no disponible")
    
    return edge_model.get_model_info()


@router.post("/edge/predict", response_model=EdgePredictionResponse)
async def edge_predict(file: UploadFile = File(...)):
    """
    Predicción con modelo edge (rápido, ligero)
    - Tamaño: ~5-10 MB
    - Latencia: 200-500ms
    - Ideal para dispositivos móviles
    """
    if not edge_model:
        raise HTTPException(status_code=503, detail="Modelo edge no disponible")
    
    try:
        import time
        image_bytes = await file.read()
        
        start_time = time.time()
        result = edge_model.predict(image_bytes)
        inference_time = (time.time() - start_time) * 1000
        
        return EdgePredictionResponse(
            label=result['label'],
            confidence=result['confidence'],
            inference_time_ms=inference_time,
            device=edge_model.device
        )
    
    except Exception as e:
        logger.error(f"Error en edge prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/edge/benchmark")
async def edge_benchmark():
    """Benchmark de rendimiento del modelo edge"""
    if not edge_model:
        raise HTTPException(status_code=503, detail="Modelo edge no disponible")
    
    from app.services.inference_edge import EdgeOptimizer
    
    benchmark = EdgeOptimizer.benchmark(edge_model, num_iterations=50)
    recommendations = EdgeOptimizer.get_optimization_recommendations(
        edge_model.get_model_info()
    )
    
    return {
        "benchmark": benchmark,
        "recommendations": recommendations
    }


@router.post("/edge/export")
async def export_edge_model(format: str = "onnx"):
    """
    Exportar modelo edge a diferentes formatos
    - onnx: Compatible con TensorFlow Lite, Core ML, etc.
    - torchscript: Ejecutable sin PyTorch
    """
    if not edge_model:
        raise HTTPException(status_code=503, detail="Modelo edge no disponible")
    
    try:
        if format == "onnx":
            output_path = "models/edge_model.onnx"
            edge_model.export_onnx(output_path)
        elif format == "torchscript":
            output_path = "models/edge_model.pt"
            edge_model.export_torchscript(output_path)
        else:
            raise HTTPException(status_code=400, detail=f"Formato desconocido: {format}")
        
        return {
            "status": "ok",
            "format": format,
            "output_path": output_path,
            "message": f"Modelo exportado a {format}"
        }
    
    except Exception as e:
        logger.error(f"Error en exportación: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# ENDPOINTS: FINE-TUNING
# ============================================================================

@router.post("/finetune/status")
async def finetune_status():
    """Estado de jobs de fine-tuning"""
    return {
        "status": "ready",
        "models_directory": "finetuned_models",
        "message": "Sistema listo para fine-tuning"
    }


@router.post("/finetune/train")
async def finetune_train(
    background_tasks: BackgroundTasks,
    config: FineTuningConfig
):
    """
    Iniciar fine-tuning con datos personalizados
    
    Requiere:
    - Directorio: /datasets/custom/images/ (imágenes)
    - Archivo: /datasets/custom/annotations.json (etiquetas)
    
    Formato annotations.json:
    {
        "image1.jpg": "plant",
        "image2.jpg": "dry_flower",
        ...
    }
    """
    try:
        from app.services.fine_tuning import FineTuningPipeline
        from app.services.inference_optimized import get_inference_engine
        
        model = get_inference_engine().model
        
        pipeline = FineTuningPipeline(
            model,
            dataset_dir="datasets/custom"
        )
        
        # Ejecutar en background
        background_tasks.add_task(
            pipeline.run,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate
        )
        
        return {
            "status": "started",
            "job_id": "finetune_001",
            "config": config.dict(),
            "message": "Fine-tuning iniciado en background"
        }
    
    except Exception as e:
        logger.error(f"Error iniciando fine-tuning: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/finetune/models")
async def list_finetuned_models():
    """Listar modelos fine-tuned disponibles"""
    from pathlib import Path
    import json
    
    finetuned_dir = Path("finetuned_models")
    
    if not finetuned_dir.exists():
        return {"models": []}
    
    models = []
    for metadata_file in finetuned_dir.glob("metadata_*.json"):
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            model_file = metadata_file.parent / f"finetuned_{metadata['timestamp']}.pt"
            
            models.append({
                "timestamp": metadata['timestamp'],
                "epochs": metadata['epochs'],
                "best_val_accuracy": metadata['best_val_accuracy'],
                "model_path": str(model_file),
                "metadata_path": str(metadata_file)
            })
        except:
            pass
    
    return {
        "total": len(models),
        "models": sorted(models, key=lambda x: x['timestamp'], reverse=True)
    }


# ============================================================================
# ENDPOINTS: UTILIDADES
# ============================================================================

@router.get("/health/advanced")
async def health_advanced():
    """Health check avanzado"""
    return {
        "cache_available": cache_manager is not None,
        "explainability_available": explainability_engine is not None,
        "edge_model_available": edge_model is not None,
        "finetuning_available": True,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }


@router.get("/features")
async def list_features():
    """Listar características avanzadas disponibles"""
    return {
        "features": [
            {
                "name": "Caché Inteligente",
                "endpoint": "/api/v1/advanced/cache/stats",
                "description": "Sistema multinivel de caché con Redis, disco y memoria"
            },
            {
                "name": "Explicabilidad GRAD-CAM",
                "endpoint": "/api/v1/advanced/explain",
                "description": "Heatmaps de activación para explicar predicciones"
            },
            {
                "name": "Modelo Edge",
                "endpoint": "/api/v1/advanced/edge/predict",
                "description": "Modelo ligero para dispositivos móviles (5-10MB)"
            },
            {
                "name": "Fine-tuning",
                "endpoint": "/api/v1/advanced/finetune/train",
                "description": "Entrenar modelo con datos personalizados"
            },
            {
                "name": "Análisis de Confianza",
                "endpoint": "/api/v1/advanced/explain",
                "description": "Análisis adaptativo de confianza con entropía y margen"
            }
        ]
    }
