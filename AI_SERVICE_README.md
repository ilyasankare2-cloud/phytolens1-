# PhytoLens - Sistema de IA para Clasificación de Plantas

## 🎯 Descripción

PhytoLens es un sistema de análisis de imágenes basado en IA que utiliza **EfficientNetV2-S** para clasificar plantas y derivados en 5 categorías:

- 🌿 **plant** - Planta viva
- 🌾 **dry_flower** - Flor seca
- 💎 **resin** - Resina
- 🧪 **extract** - Extracto
- 🏭 **processed** - Procesado

## 🚀 Instalación

### Requisitos previos
- Python 3.8+
- CUDA 11.8+ (opcional, para GPU)
- pip o conda

### Instalación de dependencias

```bash
# Instalar dependencias principales
pip install -r requirements.txt

# Instalar dependencias de IA
pip install -r requirements_ai.txt
```

## 📚 Endpoints de la API

### 1. Analizar Imagen (POST)
```http
POST /api/v1/scans/analyze
Content-Type: multipart/form-data

file: <image_file>
```

**Respuesta exitosa (200)**:
```json
{
  "id": 1,
  "user_id": 1,
  "image_path": "uploads/1_1705315800.0_photo.jpg",
  "result": {
    "label": "plant",
    "confidence": 0.9532,
    "all_probabilities": {
      "plant": 0.9532,
      "dry_flower": 0.0312,
      "resin": 0.0098,
      "extract": 0.0038,
      "processed": 0.0020
    }
  },
  "created_at": "2026-01-15T10:30:00"
}
```

### 2. Obtener Información del Modelo (GET)
```http
GET /api/v1/scans/model-info
```

**Respuesta**:
```json
{
  "device": "cuda",
  "model_name": "EfficientNetV2-S",
  "num_classes": 5,
  "classes": ["plant", "dry_flower", "resin", "extract", "processed"],
  "image_size": 224
}
```

### 3. Obtener Escaneos del Usuario (GET)
```http
GET /api/v1/scans/?skip=0&limit=100
```

### 4. Obtener Escaneo Específico (GET)
```http
GET /api/v1/scans/{scan_id}
```

### 5. Eliminar Escaneo (DELETE)
```http
DELETE /api/v1/scans/{scan_id}
```

## 🔧 Uso en Código

### Uso básico del motor de inferencia

```python
from app.services.inference import get_inference_engine

# Obtener instancia del motor
engine = get_inference_engine()

# Predicción desde bytes
with open("image.jpg", "rb") as f:
    image_bytes = f.read()

result = engine.predict(image_bytes)
print(f"Clasificación: {result['label']}")
print(f"Confianza: {result['confidence']:.2%}")
```

### Predicción en lote

```python
from app.services.inference import get_inference_engine

engine = get_inference_engine()

# Procesar múltiples imágenes
images = [open(f"img{i}.jpg", "rb").read() for i in range(5)]
results = engine.predict_batch(images)

for result in results:
    print(f"{result['label']}: {result['confidence']:.2%}")
```

### Predicción desde archivo

```python
engine = get_inference_engine()
result = engine.predict_from_path("path/to/image.jpg")
```

## 🎓 Detalles Técnicos

### Arquitectura del Modelo

- **Backbone**: EfficientNetV2-S preentrenado en ImageNet
- **Cabeza clasificadora**: Capa lineal (1280 → 5 clases)
- **Regularización**: Dropout (0.3)
- **Normalización**: ImageNet (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])

### Preprocesamiento de Imágenes

1. Redimensionar a 256×256
2. Recorte central a 224×224
3. Conversión a tensor
4. Normalización ImageNet
5. Batch aggregation

### Dispositivo Automático

El sistema detecta automáticamente:
- GPU NVIDIA (CUDA)
- CPU como fallback

## 📊 Métricas de Rendimiento

| Métrica | Valor |
|---------|-------|
| Tamaño del modelo | ~84 MB |
| Tiempo de inferencia (GPU) | ~50-100ms |
| Tiempo de inferencia (CPU) | ~500-1000ms |
| Memoria RAM (inferencia) | ~500 MB |
| Memoria VRAM (GPU) | ~1-2 GB |

## 🔐 Seguridad

- ✅ Validación de tipos de archivo
- ✅ Límite de tamaño de imagen
- ✅ Autenticación de usuario requerida
- ✅ Aislamiento de datos por usuario
- ✅ Gestión automática de recursos

## 🐛 Troubleshooting

### Error: "CUDA out of memory"
```python
# Usar CPU en lugar de GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

### Error: "Model file not found"
```python
# El modelo se descargará automáticamente en la primera ejecución
# Esperar a que se complete la descarga (~500 MB)
```

### Imágenes no reconocidas
- Asegurar que la imagen sea clara y bien iluminada
- Probar con diferentes ángulos
- Verificar que el archivo sea PNG, JPEG o WebP válido

## 📝 Logging

Los logs se guardan con la siguiente información:
- Carga del modelo
- Inferencias completadas
- Errores y excepciones
- Información del dispositivo

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 🚀 Próximas Mejoras

- [ ] Fine-tuning con datos específicos
- [ ] Modelo multi-label
- [ ] Explicabilidad (GRAD-CAM)
- [ ] Caché de predicciones
- [ ] Análisis de confianza adaptativo
- [ ] Modelo más ligero para edge devices

## 📄 Licencia

Proprietario - PhytoLens 2026

## 👨‍💻 Autor

Desarrollado para PhytoLens Backend - Enero 2026
