"""
📱 Modelo ligero para dispositivos de borde (Edge)
MobileNetV3 Small: ~2.5M parámetros, 5MB
"""

import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import numpy as np
import logging
from typing import Dict, Tuple, Optional
import io

logger = logging.getLogger(__name__)


class VisionPlantEdgeModel:
    """
    Modelo ligero optimizado para edge devices
    - Tamaño: ~5-10 MB
    - Memoria: ~100-150 MB
    - Latencia: 200-500ms en CPU
    """
    
    def __init__(self, num_classes: int = 5, quantize: bool = False):
        self.num_classes = num_classes
        self.quantize = quantize
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Cargar MobileNetV3-Small preentrenado
        self.model = models.mobilenet_v3_small(
            pretrained=True,
            progress=True
        )
        
        # Reemplazar cabeza clasificadora
        num_features = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Quantización (reduce tamaño a ~2.5MB)
        if quantize:
            self._quantize_model()
        
        logger.info(
            f"✓ Modelo Edge cargado: "
            f"device={self.device}, "
            f"quantized={quantize}"
        )
    
    def _quantize_model(self):
        """Aplicar quantización INT8"""
        try:
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(self.model, inplace=True)
            torch.quantization.convert(self.model, inplace=True)
            logger.info("✓ Modelo quantizado a INT8")
        except Exception as e:
            logger.warning(f"⚠️ Quantización no disponible: {e}")
    
    def predict(self, image_bytes: bytes) -> Dict:
        """Predicción rápida"""
        try:
            # Cargar imagen
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Preprocesar
            image_tensor = self._preprocess(image)
            
            # Predicción
            with torch.no_grad():
                output = self.model(image_tensor)
                probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            
            # Resultado
            class_idx = np.argmax(probs)
            class_names = ['plant', 'dry_flower', 'resin', 'extract', 'processed']
            
            return {
                'label': class_names[class_idx],
                'confidence': float(probs[class_idx]),
                'all_probabilities': {
                    class_names[i]: float(probs[i])
                    for i in range(len(class_names))
                }
            }
        except Exception as e:
            logger.error(f"Error en predicción edge: {e}")
            raise
    
    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocesar imagen"""
        # Redimensionar
        image = image.resize((256, 256), Image.LANCZOS)
        
        # Recorte central
        w, h = image.size
        crop_size = min(w, h)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        image = image.crop((left, top, left + crop_size, top + crop_size))
        image = image.resize((224, 224), Image.LANCZOS)
        
        # Tensor
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Normalización ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        # CHW
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
        return img_tensor.unsqueeze(0).to(self.device)
    
    def get_model_info(self) -> Dict:
        """Información del modelo"""
        # Contar parámetros
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        return {
            'name': 'MobileNetV3-Small',
            'architecture': 'Lightweight CNN',
            'parameters': {
                'total': total_params,
                'trainable': trainable_params,
                'millions': total_params / 1e6
            },
            'input_size': 224,
            'num_classes': self.num_classes,
            'quantized': self.quantize,
            'device': self.device,
            'estimated_size_mb': (total_params * 4) / (1024 * 1024)  # FP32
        }
    
    def export_onnx(self, output_path: str = "model_edge.onnx"):
        """Exportar a ONNX para mayor compatibilidad"""
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                input_names=['image'],
                output_names=['logits'],
                opset_version=12,
                do_constant_folding=True
            )
            logger.info(f"✓ Modelo exportado a ONNX: {output_path}")
        except Exception as e:
            logger.error(f"Error exportando ONNX: {e}")
    
    def export_torchscript(self, output_path: str = "model_edge.pt"):
        """Exportar a TorchScript para inferencia sin dependencias"""
        try:
            scripted_model = torch.jit.script(self.model)
            scripted_model.save(output_path)
            logger.info(f"✓ Modelo exportado a TorchScript: {output_path}")
        except Exception as e:
            logger.error(f"Error exportando TorchScript: {e}")


class EdgeOptimizer:
    """Herramientas de optimización para edge deployment"""
    
    @staticmethod
    def benchmark(model: VisionPlantEdgeModel, num_iterations: int = 100) -> Dict:
        """Benchmark de rendimiento"""
        import time
        
        dummy_input = torch.randn(1, 3, 224, 224)
        if model.device == 'cuda':
            dummy_input = dummy_input.cuda()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model.model(dummy_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.time()
                _ = model.model(dummy_input)
                times.append(time.time() - start)
        
        times = np.array(times)
        
        return {
            'device': model.device,
            'mean_time_ms': float(times.mean() * 1000),
            'std_time_ms': float(times.std() * 1000),
            'min_time_ms': float(times.min() * 1000),
            'max_time_ms': float(times.max() * 1000),
            'throughput_images_per_second': float(1 / times.mean()),
            'iterations': num_iterations
        }
    
    @staticmethod
    def get_optimization_recommendations(model_info: Dict) -> list:
        """Recomendaciones de optimización"""
        recommendations = []
        
        params_millions = model_info['parameters']['millions']
        
        if params_millions > 10:
            recommendations.append(
                "📉 Model es grande para edge (>10M params). "
                "Considerar quantización o destilación."
            )
        
        if not model_info['quantized']:
            recommendations.append(
                "🔢 Quantización puede reducir tamaño 75% sin perder precisión."
            )
        
        if model_info['estimated_size_mb'] > 20:
            recommendations.append(
                "💾 Tamaño modelo es >20MB. Para mobile, usar versión quantizada."
            )
        
        return recommendations
