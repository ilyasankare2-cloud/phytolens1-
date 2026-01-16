# PhytoLens - Sistema de IA para Clasificación de Plantas

## 📋 Descripción

PhytoLens es un sistema de análisis de imágenes que utiliza **EfficientNetV2-S** para clasificar plantas y derivados en 5 categorías:

- 🌿 **plant** - Planta viva
- 🌾 **dry_flower** - Flor seca
- 💎 **resin** - Resina
- 🧪 **extract** - Extracto
- 🏭 **processed** - Procesado

## 🎯 Características

- ✅ Modelo de IA preentrenado (EfficientNetV2-S)
- ✅ API REST completa con FastAPI
- ✅ Soporte automático GPU/CPU
- ✅ Documentación Swagger interactiva
- ✅ Predicciones individuales y en lote
- ✅ Validación de imágenes
- ✅ Logging completo
- ✅ Endpoints CORS habilitados

## 🚀 Inicio Rápido

### 1. Instalación

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/phytolens.git
cd phytolens/backend

# Crear entorno virtual (recomendado)
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
pip install -r requirements_ai.txt
```

### 2. Ejecutar Servidor

```bash
python simple_server.py
```

Servidor disponible en: `http://127.0.0.1:8001`

### 3. Acceder a Documentación

- **Swagger UI**: http://127.0.0.1:8001/docs
- **ReDoc**: http://127.0.0.1:8001/redoc

## 📚 API Endpoints

### Health Check
```bash
GET /health
```

### Información del Modelo
```bash
GET /model-info
```

### Analizar Imagen
```bash
POST /analyze
Content-Type: multipart/form-data

file: <imagen.jpg>
```

**Respuesta:**
```json
{
  "success": true,
  "timestamp": "2026-01-15T10:30:00",
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
  "message": "Imagen analizada correctamente. Clase detectada: plant"
}
```

### Analizar Múltiples Imágenes
```bash
POST /analyze-batch
Content-Type: multipart/form-data

files: <imagen1.jpg>, <imagen2.jpg>, <imagen3.jpg>
```

## 💻 Uso Programático

```python
from app.services.inference import get_inference_engine

# Obtener motor
engine = get_inference_engine()

# Predicción desde bytes
with open("imagen.jpg", "rb") as f:
    result = engine.predict(f.read())

print(f"Clase: {result['label']}")
print(f"Confianza: {result['confidence']:.2%}")

# Predicción en lote
images = [open(f"img{i}.jpg", "rb").read() for i in range(3)]
results = engine.predict_batch(images)
```

## 📊 Requisitos Técnicos

| Componente | Versión |
|-----------|---------|
| Python | 3.8+ |
| PyTorch | 2.0+ |
| FastAPI | 0.100+ |
| CUDA | 11.8+ (opcional) |

## 🗂️ Estructura del Proyecto

```
phytolens/backend/
├── app/
│   ├── main.py                    # Aplicación FastAPI
│   ├── api/
│   │   ├── v1/
│   │   │   └── endpoints/
│   │   │       └── scans.py       # Endpoints de análisis
│   │   └── deps.py                # Dependencias
│   ├── services/
│   │   └── inference.py           # Motor de IA
│   ├── schemas/
│   │   └── scan.py                # Esquemas Pydantic
│   ├── models/
│   │   ├── scan.py                # Modelo BD
│   │   └── user.py
│   └── core/
│       └── config.py              # Configuración
├── simple_server.py               # Servidor simple
├── test_ia_execution.py           # Pruebas IA
└── requirements.txt               # Dependencias
```

## 🔧 Configuración

### Variables de Entorno (.env)

```bash
# Base de Datos
POSTGRES_SERVER=localhost
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=phytolens

# JWT
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
BACKEND_CORS_ORIGINS=["http://localhost:3000"]
```

## 🧪 Pruebas

```bash
# Pruebas del sistema completo
python test_final.py

# Pruebas de IA
python test_ia_execution.py

# Cliente HTTP
python test_client.py
```

## 📈 Rendimiento

| Métrica | Valor |
|---------|-------|
| Tamaño modelo | ~84 MB |
| Tiempo inferencia (GPU) | ~50-100ms |
| Tiempo inferencia (CPU) | ~500-1000ms |
| Memoria entrada | ~500 MB |

## 🔐 Seguridad

- ✅ Validación de tipos de archivo
- ✅ Límite de tamaño de imagen
- ✅ Autenticación JWT
- ✅ CORS configurable
- ✅ Variables de entorno sensibles

## 🐛 Troubleshooting

### CUDA out of memory
```python
# Usar CPU en lugar de GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

### Puerto 8001 en uso
```bash
# Usar otro puerto
python -m uvicorn app.main:app --port 8002
```

### Modelo no descargado
- Primera ejecución descargará ~500 MB
- Asegúrate de tener conexión a internet

## 🚀 Despliegue Producción

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements_ai.txt ./
RUN pip install -r requirements.txt -r requirements_ai.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - POSTGRES_SERVER=db
      - POSTGRES_DB=phytolens
    depends_on:
      - db
  
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=phytolens
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

## 📝 Licencia

Proprietario - PhytoLens 2026

## 👨‍💻 Autor

Desarrollado en enero de 2026

## 📞 Soporte

Para reportar problemas o sugerencias, abre un issue en GitHub.

---

**¡Gracias por usar PhytoLens!** 🌿
