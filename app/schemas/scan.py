from datetime import datetime
from typing import Optional, Dict
from pydantic import BaseModel, Field


class ScanResultBase(BaseModel):
    """Resultado base de predicción"""
    label: str = Field(..., description="Etiqueta de clasificación")
    confidence: float = Field(..., ge=0, le=1, description="Confianza de la predicción")
    all_probabilities: Dict[str, float] = Field(..., description="Probabilidades de todas las clases")



class QualityMetrics(BaseModel):
    """Métricas de calidad de imagen y muestra"""
    visual_score: float = Field(..., ge=0, le=10, description="Puntuación visual general 0-10")
    sharpness: float = Field(..., description="Nivel de nitidez de la imagen")
    brightness: float = Field(..., description="Nivel de brillo/exposición")
    contrast: float = Field(..., description="Nivel de contraste")
    is_valid: bool = Field(True, description="Si la imagen es válida para análisis profesional")
    rejection_reason: Optional[str] = None


class ChemicalProfile(BaseModel):
    """Estimación de perfil químico (Beta)"""
    thc_range: str = Field(..., description="Rango estimado de THC (ej: '15-20%')")
    cbd_range: str = Field(..., description="Rango estimado de CBD")
    terpenes_primary: Optional[str] = Field(None, description="Terpenos probables")
    potency_category: str = Field(..., description="Categoría de potencia: Baja, Media, Alta, Elite")


class AnalysisMeta(BaseModel):
    """Metadatos técnicos del análisis"""
    inference_time_ms: float
    model_version: str
    device_used: str
    feature_flags: list[str]
    input_resolution: str


class ScanResult(ScanResultBase):
    """Resultado de predicción profesional completo"""
    label_id: Optional[int] = Field(None, description="ID numérico de la clase")
    
    # Nuevos campos V3
    certainty_score: float = Field(..., description="Puntuación de certidumbre del sistema (Trust Score)")
    quality: Optional[QualityMetrics] = None
    chemistry: Optional[ChemicalProfile] = None
    meta: Optional[AnalysisMeta] = None
    
    # Backward compatibility
    top_3_predictions: list[Dict[str, float]] = []



class ScanBase(BaseModel):
    """Base para modelos de escaneo"""
    pass


class ScanCreate(ScanBase):
    """Crear un escaneo (metadatos)"""
    pass


class ScanResponse(ScanBase):
    """Respuesta de escaneo"""
    id: int
    user_id: int
    image_path: str
    result: ScanResult
    created_at: datetime

    class Config:
        orm_mode = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "user_id": 1,
                "image_path": "uploads/1_image.jpg",
                "result": {
                    "label": "plant",
                    "confidence": 0.95,
                    "all_probabilities": {
                        "plant": 0.95,
                        "dry_flower": 0.03,
                        "resin": 0.01,
                        "extract": 0.005,
                        "processed": 0.005
                    }
                },
                "created_at": "2026-01-15T10:30:00"
            }
        }


class Scan(ScanBase):
    """Modelo de Escaneo (desde BD)"""
    id: int
    user_id: int
    image_path: str
    result_json: Optional[ScanResult] = None
    primary_classification: Optional[str] = None
    confidence_score: Optional[float] = None
    created_at: datetime

    class Config:
        orm_mode = True
