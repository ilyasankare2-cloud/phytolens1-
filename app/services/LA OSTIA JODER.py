import shutil
import os
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.orm import Session
from app.api import deps
from app.models.user import User
from app.models.scan import Scan
from app.schemas.scan import Scan as ScanSchema, ScanResult, ScanResponse
from app.services.inference import get_inference_engine

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


@router.post("/analyze", response_model=ScanResponse)
async def analyze_plant_image(
    *,
    db: Session = Depends(deps.get_db),
    file: UploadFile = File(...),
    current_user: User = Depends(deps.get_current_user)
):
    """
    Analizar una imagen de planta y obtener clasificación con IA.
    
    - **file**: Archivo de imagen (JPEG, PNG)
    - Retorna: Clasificación, confianza y probabilidades de todas las clases
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
        
        # 1. Guardar archivo
        file_location = f"{UPLOAD_DIR}/{current_user.id}_{datetime.now().timestamp()}_{file.filename}"
        os.makedirs(os.path.dirname(file_location) or '.', exist_ok=True)
        
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Archivo guardado: {file_location}")
        
        # 2. Ejecutar inferencia
        inference_engine = get_inference_engine()
        with open(file_location, "rb") as f:
            image_bytes = f.read()
        
        prediction = inference_engine.predict(image_bytes)
        logger.info(f"Predicción completada para usuario {current_user.id}: {prediction['label']}")
        
        # 3. Guardar en base de datos
        db_scan = Scan(
            user_id=current_user.id,
            image_path=file_location,
            result_json=prediction,
            primary_classification=prediction["label"],
            confidence_score=prediction["confidence"]
        )
        db.add(db_scan)
        db.commit()
        db.refresh(db_scan)
        
        return ScanResponse(
            id=db_scan.id,
            user_id=db_scan.user_id,
            image_path=db_scan.image_path,
            result=ScanResult(
                label=prediction["label"],
                confidence=prediction["confidence"],
                all_probabilities=prediction["all_probabilities"]
            ),
            created_at=db_scan.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en análisis de imagen: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al procesar la imagen"
        )


@router.get("/model-info")
async def get_model_information(current_user: User = Depends(deps.get_current_user)):
    """
    Obtener información del modelo de IA.
    """
    inference_engine = get_inference_engine()
    return inference_engine.get_model_info()

@router.get("/", response_model=list[Scan])
def read_user_scans(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
):
    """
    Obtener todos los escaneos del usuario autenticado.
    """
    scans = db.query(Scan).filter(Scan.user_id == current_user.id).offset(skip).limit(limit).all()
    return scans


@router.get("/{scan_id}", response_model=ScanSchema)
def read_scan(
    scan_id: int,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
):
    """
    Obtener detalles de un escaneo específico.
    """
    scan = db.query(Scan).filter(
        Scan.id == scan_id,
        Scan.user_id == current_user.id
    ).first()
    
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Escaneo no encontrado"
        )
    return scan


@router.delete("/{scan_id}")
def delete_scan(
    scan_id: int,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
):
    """
    Eliminar un escaneo.
    """
    scan = db.query(Scan).filter(
        Scan.id == scan_id,
        Scan.user_id == current_user.id
    ).first()
    
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Escaneo no encontrado"
        )
    
    # Eliminar archivo si existe
    if os.path.exists(scan.image_path):
        try:
            os.remove(scan.image_path)
        except Exception as e:
            logger.warning(f"Error al eliminar archivo {scan.image_path}: {e}")
    
    db.delete(scan)
    db.commit()
    
    return {"message": "Escaneo eliminado correctamente"}
