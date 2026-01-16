"""
Servicio de Inferencia de IA para PhytoLens - Versión Mejorada
Utiliza EfficientNetV2-M con arquitectura avanzada para mejor precisión
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

logger = logging.getLogger(__name__)


class PhytoClassifierV2(nn.Module):
    """
    Modelo mejorado de clasificación de plantas basado en EfficientNetV2-M.
    Incluye arquitectura multi-capa con regularización avanzada.
    
    Mejoras:
    - Backbone más potente (V2-M)
    - Capas densas adicionales
    - Batch normalization
    - Dropout estratégico
    - Skip connections conceptual
    """
    def __init__(self, num_classes: int = 5, dropout_rate: float = 0.4):
        super(PhytoClassifierV2, self).__init__()
        self.num_classes = num_classes
        
        # Cargar EfficientNetV2-M preentrenado (más potente que S)
        self.backbone = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
        
        # Obtener número de features del backbone
        in_features = self.backbone.classifier[1].in_features
        
        # Head clasificador mejorado
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
        
        # Capas adicionales para mejor generalización
        self.dropout = nn.Dropout(dropout_rate)
        
        # Inicializar pesos
        self._init_weights()
    
    def _init_weights(self):
        """Inicializar pesos de capas personalizadas"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass del modelo"""
        # Extraer features del backbone
        features = self.backbone.features(x)
        x = self.backbone.avgpool(features)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.backbone.classifier(x)
        return x


class InferenceEngineV2:
    """
    Motor de inferencia mejorado para predicciones de clasificación de plantas.
    
    Mejoras:
    - Test-Time Augmentation (TTA)
    - Caché de predicciones
    - Mejor preprocesamiento
    - Validación robusta de entrada
    - Confidence scoring mejorado
    """
    
    # Clases de clasificación
    CLASS_LABELS = {
        0: 'plant',
        1: 'dry_flower',
        2: 'resin',
        3: 'extract',
        4: 'processed'
    }
    
    # Configuración de imágenes
    IMAGE_SIZE = 384  # Aumentado de 224 para mejor precisión
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    def __init__(self, model_path: Optional[str] = None, use_tta: bool = True):
        """
        Inicializar el motor de inferencia.
        
        Args:
            model_path: Ruta al archivo de pesos del modelo
            use_tta: Usar Test-Time Augmentation para mejor precisión
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {self.device}")
        
        self.use_tta = use_tta
        logger.info(f"Test-Time Augmentation: {'Habilitado' if use_tta else 'Deshabilitado'}")
        
        # Cargar modelo
        self.model = PhytoClassifierV2(num_classes=len(self.CLASS_LABELS), dropout_rate=0.4)
        
        # Cargar pesos si se proporciona una ruta
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                logger.info(f"Modelo cargado desde: {model_path}")
            except Exception as e:
                logger.warning(f"No se pudo cargar el modelo desde {model_path}: {e}. Usando modelo preentrenado.")
        else:
            logger.info("Usando modelo EfficientNetV2-M preentrenado")
        
        # Mover modelo a dispositivo y modo evaluación
        self.model.to(self.device)
        self.model.eval()
        
        # Configurar transformaciones principales
        self.transform_main = transforms.Compose([
            transforms.Resize((self.IMAGE_SIZE + 32, self.IMAGE_SIZE + 32)),
            transforms.CenterCrop(self.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.NORMALIZE_MEAN,
                std=self.NORMALIZE_STD
            )
        ])
        
        # Transformaciones para TTA (variaciones de la imagen)
        self.transforms_tta = [
            transforms.Compose([
                transforms.Resize((self.IMAGE_SIZE + 32, self.IMAGE_SIZE + 32)),
                transforms.CenterCrop(self.IMAGE_SIZE),
                transforms.RandomHorizontalFlip(p=1.0),  # Flip horizontal
                transforms.ToTensor(),
                transforms.Normalize(self.NORMALIZE_MEAN, self.NORMALIZE_STD)
            ]),
            transforms.Compose([
                transforms.Resize((self.IMAGE_SIZE + 32, self.IMAGE_SIZE + 32)),
                transforms.RandomCrop(self.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(self.NORMALIZE_MEAN, self.NORMALIZE_STD)
            ]),
            transforms.Compose([
                transforms.Resize((self.IMAGE_SIZE + 32, self.IMAGE_SIZE + 32)),
                transforms.CenterCrop(self.IMAGE_SIZE),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(self.NORMALIZE_MEAN, self.NORMALIZE_STD)
            ]),
        ]
        
        # Caché para predicciones
        self.prediction_cache = {}
        logger.info("Motor de inferencia inicializado con mejoras")
    
    def _get_cache_key(self, image_bytes: bytes) -> str:
        """Generar clave de caché basada en hash de imagen"""
        return hashlib.md5(image_bytes).hexdigest()
    
    def _validate_image(self, image: Image.Image) -> bool:
        """Validar que la imagen sea válida"""
        try:
            # Validar tamaño mínimo
            if image.size[0] < 100 or image.size[1] < 100:
                logger.warning(f"Imagen muy pequeña: {image.size}")
                return False
            
            # Validar modo
            if image.mode not in ['RGB', 'L', 'RGBA']:
                logger.warning(f"Modo de imagen no soportado: {image.mode}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validando imagen: {e}")
            return False
    
    def predict(self, image_bytes: bytes) -> Dict:
        """
        Realizar predicción en una imagen con validación y caché.
        
        Args:
            image_bytes: Bytes de la imagen a clasificar
            
        Returns:
            Diccionario con resultados de la predicción mejorados
        """
        try:
            # Verificar caché
            cache_key = self._get_cache_key(image_bytes)
            if cache_key in self.prediction_cache:
                logger.debug(f"Predicción obtenida del caché")
                return self.prediction_cache[cache_key]
            
            # Cargar y validar imagen
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            if not self._validate_image(image):
                raise ValueError("Imagen no válida o demasiado pequeña")
            
            # Realizar predicción
            if self.use_tta:
                result = self._predict_with_tta(image)
            else:
                result = self._predict_single(image)
            
            # Guardar en caché
            self.prediction_cache[cache_key] = result
            
            logger.info(f"Predicción realizada: {result['label']} (confianza: {result['confidence']:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"Error durante la predicción: {e}")
            raise
    
    def _predict_single(self, image: Image.Image) -> Dict:
        """Predicción simple sin TTA"""
        tensor = self.transform_main(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)
        
        predicted_idx = predicted_idx.item()
        confidence = confidence.item()
        
        return self._build_result(predicted_idx, probs[0], confidence)
    
    def _predict_with_tta(self, image: Image.Image) -> Dict:
        """Predicción con Test-Time Augmentation para mayor precisión"""
        all_probs = []
        
        # Predicción principal
        tensor = self.transform_main(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            all_probs.append(F.softmax(outputs, dim=1))
        
        # Predicciones con augmentaciones
        for transform in self.transforms_tta:
            try:
                tensor = transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.model(tensor)
                    all_probs.append(F.softmax(outputs, dim=1))
            except Exception as e:
                logger.debug(f"Error en TTA: {e}")
                continue
        
        # Promediar probabilidades (ensemble)
        avg_probs = torch.stack(all_probs).mean(dim=0)[0]
        confidence, predicted_idx = torch.max(avg_probs, 0)
        
        return self._build_result(predicted_idx.item(), avg_probs, confidence.item())
    
    def _build_result(self, predicted_idx: int, probs: torch.Tensor, confidence: float) -> Dict:
        """Construir resultado con información extendida"""
        # Convertir a CPU si es necesario
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()
        
        # Calcular métricas adicionales
        all_probs_dict = {
            self.CLASS_LABELS[i]: round(float(probs[i]), 4)
            for i in range(len(self.CLASS_LABELS))
        }
        
        # Top-3 predicciones
        sorted_classes = sorted(all_probs_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Calcular "certainty" (diferencia entre top-1 y top-2)
        certainty = sorted_classes[0][1] - sorted_classes[1][1] if len(sorted_classes) > 1 else 1.0
        
        result = {
            "label": self.CLASS_LABELS[predicted_idx],
            "label_id": predicted_idx,
            "confidence": round(confidence, 4),
            "certainty": round(certainty, 4),  # Nueva métrica
            "all_probabilities": all_probs_dict,
            "top_3_predictions": [
                {"label": label, "probability": prob}
                for label, prob in sorted_classes[:3]
            ],
            "model_version": "V2-M with TTA" if self.use_tta else "V2-M"
        }
        
        return result
    
    def predict_batch(self, image_list: List[bytes]) -> List[Dict]:
        """
        Realizar predicciones en lote optimizado.
        
        Args:
            image_list: Lista de bytes de imágenes
            
        Returns:
            Lista de resultados de predicción
        """
        results = []
        for image_bytes in image_list:
            results.append(self.predict(image_bytes))
        return results
    
    def predict_from_path(self, image_path: str) -> Dict:
        """
        Realizar predicción desde una ruta de archivo.
        
        Args:
            image_path: Ruta al archivo de imagen
            
        Returns:
            Diccionario con resultados de la predicción
        """
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        return self.predict(image_bytes)
    
    def get_model_info(self) -> Dict:
        """Obtener información del modelo mejorada"""
        return {
            "device": str(self.device),
            "model_name": "EfficientNetV2-M",
            "model_version": "V2 Improved",
            "num_classes": len(self.CLASS_LABELS),
            "classes": list(self.CLASS_LABELS.values()),
            "image_size": self.IMAGE_SIZE,
            "tta_enabled": self.use_tta,
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
        """Limpiar caché de predicciones"""
        self.prediction_cache.clear()
        logger.info("Caché de predicciones limpiado")


# Instancia global del motor de inferencia (versión mejorada)
_inference_engine: Optional[InferenceEngineV2] = None


def get_inference_engine(use_tta: bool = True) -> InferenceEngineV2:
    """
    Obtener la instancia global del motor de inferencia mejorado.
    
    Args:
        use_tta: Usar Test-Time Augmentation
    
    Returns:
        Instancia del motor de inferencia
    """
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = InferenceEngineV2(use_tta=use_tta)
        logger.info("Motor de inferencia V2 inicializado")
    return _inference_engine


def initialize_inference_engine(model_path: Optional[str] = None, use_tta: bool = True) -> InferenceEngineV2:
    """
    Inicializar el motor de inferencia mejorado de forma explícita.
    
    Args:
        model_path: Ruta al archivo de pesos del modelo
        use_tta: Usar Test-Time Augmentation
        
    Returns:
        Instancia del motor de inferencia
    """
    global _inference_engine
    _inference_engine = InferenceEngineV2(model_path=model_path, use_tta=use_tta)
    return _inference_engine
