"""
Motor de Inferencia Optimizado para VisionPlant
Máxima velocidad, precisión y confiabilidad
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from PIL import Image
import io
import logging
from typing import Dict, Optional, List
from functools import lru_cache
import hashlib
import time

logger = logging.getLogger(__name__)

# Usar CPU siempre (más confiable para servidor)
DEVICE = torch.device('cpu')

class VisionPlantClassifier(nn.Module):
    """
    Clasificador optimizado para VisionPlant
    Máxima precisión y velocidad
    """
    def __init__(self, num_classes: int = 5, dropout_rate: float = 0.4):
        super(VisionPlantClassifier, self).__init__()
        self.num_classes = num_classes
        
        # Cargar EfficientNetV2-M preentrenado
        self.backbone = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        
        # Head optimizado
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            
            nn.Linear(128, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()
        
        # Modo evaluación permanente
        self.eval()
    
    def _init_weights(self):
        """Inicialización Kaiming"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features(x)
        x = self.backbone.avgpool(features)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.backbone.classifier(x)
        return x


class OptimizedInferenceEngine:
    """Motor de inferencia optimizado"""
    
    def __init__(self, use_tta: bool = False, cache_size: int = 256):
        self.device = DEVICE
        self.use_tta = use_tta
        self.model = VisionPlantClassifier().to(self.device)
        self.prediction_cache = {}
        self.cache_size = cache_size
        self.class_names = ['plant', 'dry_flower', 'resin', 'extract', 'processed']
        
        # Transformaciones estándar (sin data augmentation)
        self.transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Transformaciones para TTA
        self.tta_transforms = [
            transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ]
        
        logger.info("✓ Motor de inferencia inicializado")
    
    def _get_cache_key(self, image_bytes: bytes) -> str:
        """Generar clave de caché con MD5"""
        return hashlib.md5(image_bytes).hexdigest()
    
    def _validate_image(self, image: Image.Image) -> bool:
        """Validar imagen"""
        try:
            if image.size[0] < 50 or image.size[1] < 50:
                logger.warning(f"Imagen muy pequeña: {image.size}")
                return False
            
            if image.mode not in ['RGB', 'L', 'RGBA', 'P']:
                logger.warning(f"Modo no soportado: {image.mode}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validando imagen: {e}")
            return False
    
    @torch.no_grad()
    def _predict_single(self, image: Image.Image) -> Dict:
        """Predicción simple y rápida"""
        try:
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()
            
            return self._build_result(probabilities)
        except Exception as e:
            logger.error(f"Error en predicción simple: {e}")
            raise
    
    @torch.no_grad()
    def _predict_with_tta(self, image: Image.Image) -> Dict:
        """Predicción con Test-Time Augmentation"""
        try:
            all_probs = []
            
            for transform in self.tta_transforms:
                tensor = transform(image).unsqueeze(0).to(self.device)
                outputs = self.model(tensor)
                probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
                all_probs.append(probs)
            
            # Promediar predicciones
            import numpy as np
            avg_probs = np.mean(all_probs, axis=0)
            
            return self._build_result(avg_probs)
        except Exception as e:
            logger.error(f"Error en TTA: {e}")
            raise
    
    def _build_result(self, probabilities) -> Dict:
        """Construir resultado de predicción"""
        import numpy as np
        
        # Top-1
        top_idx = np.argmax(probabilities)
        top_prob = probabilities[top_idx]
        
        # Top-2 para certeza
        top2_idx = np.argsort(probabilities)[-2]
        top2_prob = probabilities[top2_idx]
        
        # Certeza (diferencia entre top-1 y top-2)
        certainty = top_prob - top2_prob
        
        # Top-3
        top3_indices = np.argsort(probabilities)[-3:][::-1]
        top3_predictions = [
            {
                "label": self.class_names[idx],
                "probability": float(probabilities[idx])
            }
            for idx in top3_indices
        ]
        
        # Todas las probabilidades
        all_probs = {
            self.class_names[i]: float(probabilities[i])
            for i in range(len(self.class_names))
        }
        
        return {
            "label": self.class_names[top_idx],
            "confidence": float(top_prob),
            "certainty": float(certainty),
            "model_version": "VisionPlant V1.0",
            "top_3_predictions": top3_predictions,
            "all_probabilities": all_probs
        }
    
    def predict(self, image_bytes: bytes) -> Dict:
        """Predicción con caché"""
        try:
            # Verificar caché
            cache_key = self._get_cache_key(image_bytes)
            if cache_key in self.prediction_cache:
                logger.debug("Resultado obtenido del caché")
                return self.prediction_cache[cache_key]
            
            # Cargar imagen
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            if not self._validate_image(image):
                raise ValueError("Imagen no válida")
            
            # Predecir
            if self.use_tta:
                result = self._predict_with_tta(image)
            else:
                result = self._predict_single(image)
            
            # Cachear (con límite de tamaño)
            if len(self.prediction_cache) >= self.cache_size:
                # Eliminar el primer elemento (FIFO)
                self.prediction_cache.pop(next(iter(self.prediction_cache)))
            
            self.prediction_cache[cache_key] = result
            
            logger.info(f"Predicción: {result['label']} ({result['confidence']:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise
    
    def predict_batch(self, image_list: List[bytes]) -> List[Dict]:
        """Predicción en lote"""
        results = []
        for image_bytes in image_list:
            try:
                result = self.predict(image_bytes)
                results.append(result)
            except Exception as e:
                logger.error(f"Error en lote: {e}")
                results.append({"error": str(e)})
        return results
    
    def get_model_info(self) -> Dict:
        """Información del modelo"""
        return {
            "model_name": "EfficientNetV2-M",
            "model_version": "VisionPlant V1.0",
            "image_size": 384,
            "num_classes": 5,
            "class_names": self.class_names,
            "tta_enabled": self.use_tta,
            "device": str(self.device),
            "cache_size": len(self.prediction_cache),
            "features": [
                "Test-Time Augmentation",
                "Prediction Caching",
                "Advanced Head Architecture",
                "Batch Normalization",
                "Improved Validation",
                "Certainty Scoring"
            ]
        }
    
    def clear_cache(self):
        """Limpiar caché manualmente"""
        self.prediction_cache.clear()
        logger.info("Caché limpiado")


# Instancia global (lazy loading)
_engine: Optional[OptimizedInferenceEngine] = None

def get_inference_engine(use_tta: bool = False, cache_size: int = 256) -> OptimizedInferenceEngine:
    """Obtener motor de inferencia (singleton)"""
    global _engine
    
    if _engine is None:
        logger.info("Inicializando motor de inferencia...")
        _engine = OptimizedInferenceEngine(use_tta=use_tta, cache_size=cache_size)
    
    return _engine
