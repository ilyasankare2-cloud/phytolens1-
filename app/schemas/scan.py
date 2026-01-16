from datetime import datetime
from typing import Optional, Dict
from pydantic import BaseModel, Field


class ScanResultBase(BaseModel):
    """Resultado base de predicción"""
    label: str = Field(..., description="Etiqueta de clasificación")
    confidence: float = Field(..., ge=0, le=1, description="Confianza de la predicción")
    all_probabilities: Dict[str, float] = Field(..., description="Probabilidades de todas las clases")


class ScanResult(ScanResultBase):
    """Resultado de predicción completo"""
    label_id: Optional[int] = Field(None, description="ID numérico de la clase")


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
