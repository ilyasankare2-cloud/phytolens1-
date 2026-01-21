"""
🔍 Explicabilidad con GRAD-CAM
Genera heatmaps para visualizar qué partes de la imagen son importantes
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
try:
    import cv2
except ImportError:
    cv2 = None
import logging
from typing import Tuple, Optional
import base64
from io import BytesIO

logger = logging.getLogger(__name__)


class GradCAM:
    """Generador de mapas de activación por gradientes"""
    
    def __init__(self, model: torch.nn.Module, target_layer: str = 'features'):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Registrar hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Registrar forward/backward hooks"""
        def get_activations(module, input, output):
            self.activations = output.detach()
        
        def get_gradients(module, input, output):
            self.gradients = output[0].detach()
        
        # Encontrar la capa target
        for name, module in self.model.named_modules():
            if self.target_layer in name:
                module.register_forward_hook(get_activations)
                module.register_full_backward_hook(get_gradients)
                break
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int,
        device: str = 'cpu'
    ) -> np.ndarray:
        """
        Generar mapa de activación
        
        Args:
            input_tensor: Tensor de imagen (1, 3, H, W)
            class_idx: Índice de clase
            device: 'cpu' o 'cuda'
        
        Returns:
            Mapa de activación normalizado (0-255)
        """
        input_tensor = input_tensor.to(device)
        self.model.eval()
        
        # Forward pass
        with torch.enable_grad():
            output = self.model(input_tensor)
            
            # Backward pass
            self.model.zero_grad()
            target = output[0, class_idx]
            target.backward()
        
        # Calcular pesos
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = gradients.mean(dim=[1, 2])
        
        # Weighted combination
        cam = torch.zeros(activations.shape[1:], device=device)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        # Normalizar a 0-255
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = (cam * 255).astype(np.uint8)
        
        return cam
    
    @staticmethod
    def overlay_on_image(
        image_array: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Superponer heatmap en imagen"""
        # Redimensionar CAM a tamaño de imagen
        h, w = image_array.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Aplicar colormap
        heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
        
        # Convertir BGR a RGB si es necesario
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Superponer
        result = cv2.addWeighted(image_array, 1 - alpha, heatmap, alpha, 0)
        
        return result


class AdaptiveConfidence:
    """Análisis adaptativo de confianza con ajustes dinámicos"""
    
    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.class_thresholds = {i: 0.5 for i in range(num_classes)}
        self.history = []
    
    def update_threshold(self, class_idx: int, confidence: float):
        """Actualizar umbral dinámicamente"""
        self.class_thresholds[class_idx] = min(
            0.95,
            max(0.3, confidence * 0.9)  # 90% de la confianza observada
        )
    
    def analyze(
        self,
        probabilities: np.ndarray,
        model_uncertainty: Optional[float] = None
    ) -> dict:
        """
        Análisis avanzado de confianza
        
        Args:
            probabilities: Array de probabilidades (N_CLASSES,)
            model_uncertainty: Incertidumbre del modelo (opcional)
        
        Returns:
            Dict con métricas de confianza
        """
        top_idx = np.argmax(probabilities)
        top_prob = probabilities[top_idx]
        
        # Entropía (incertidumbre)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = np.log(self.num_classes)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Diferencia con segunda mejor
        sorted_probs = np.sort(probabilities)[::-1]
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
        
        # Score de confianza adaptativo
        confidence_score = (
            top_prob * 0.5 +           # 50% probabilidad
            margin * 0.3 +              # 30% margen
            (1 - normalized_entropy) * 0.2  # 20% inverso de entropía
        )
        
        # Nivel de confianza
        if confidence_score > 0.8:
            level = "muy_alta"
        elif confidence_score > 0.65:
            level = "alta"
        elif confidence_score > 0.5:
            level = "media"
        else:
            level = "baja"
        
        self.history.append({
            "confidence": confidence_score,
            "level": level
        })
        
        return {
            "confidence_score": float(confidence_score),
            "level": level,
            "top_probability": float(top_prob),
            "margin": float(margin),
            "entropy": float(entropy),
            "uncertainty": float(normalized_entropy),
            "reliable": confidence_score > self.class_thresholds.get(top_idx, 0.5)
        }


class ExplainabilityEngine:
    """Motor de explicabilidad integrado"""
    
    def __init__(self, model: torch.nn.Module, num_classes: int = 5):
        self.grad_cam = GradCAM(model)
        self.adaptive_conf = AdaptiveConfidence(num_classes)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def explain_prediction(
        self,
        image_tensor: torch.Tensor,
        probabilities: np.ndarray,
        predicted_class: int,
        original_image_array: Optional[np.ndarray] = None,
        include_heatmap: bool = True
    ) -> dict:
        """
        Generar explicación completa de predicción
        
        Args:
            image_tensor: Tensor de entrada
            probabilities: Probabilidades de clases
            predicted_class: Índice de clase predicha
            original_image_array: Imagen original para visualización
            include_heatmap: Incluir heatmap visual
        
        Returns:
            Dict con explicación completa
        """
        explanation = {
            "grad_cam": None,
            "heatmap_base64": None,
            "confidence_analysis": self.adaptive_conf.analyze(probabilities),
            "top_3": self._get_top_3(probabilities)
        }
        
        # Generar GRAD-CAM si se requiere
        if include_heatmap and original_image_array is not None:
            try:
                cam = self.grad_cam.generate(
                    image_tensor,
                    predicted_class,
                    self.device
                )
                
                # Superponer en imagen
                overlay = GradCAM.overlay_on_image(
                    original_image_array,
                    cam,
                    alpha=0.4
                )
                
                # Convertir a base64
                img_pil = Image.fromarray(overlay)
                buffer = BytesIO()
                img_pil.save(buffer, format='PNG')
                buffer.seek(0)
                explanation["heatmap_base64"] = base64.b64encode(
                    buffer.getvalue()
                ).decode()
                
            except Exception as e:
                logger.warning(f"Error generating GRAD-CAM: {e}")
        
        return explanation
    
    @staticmethod
    def _get_top_3(probabilities: np.ndarray) -> list:
        """Obtener top-3 predicciones"""
        class_names = ['plant', 'dry_flower', 'resin', 'extract', 'processed']
        top_indices = np.argsort(probabilities)[::-1][:3]
        
        return [
            {
                "class": class_names[idx],
                "probability": float(probabilities[idx]),
                "index": int(idx)
            }
            for idx in top_indices
        ]
