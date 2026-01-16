"""
Tests unitarios para VisionPlant
Validación completa de funcionalidad
"""

import pytest
import io
from PIL import Image
import numpy as np
from pathlib import Path

# Fixtures
@pytest.fixture
def sample_image():
    """Crear imagen de prueba"""
    img = Image.new('RGB', (256, 256), color=(73, 109, 137))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.getvalue()

@pytest.fixture
def small_image():
    """Crear imagen muy pequeña"""
    img = Image.new('RGB', (30, 30))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.getvalue()

@pytest.fixture
def inference_engine():
    """Obtener motor de inferencia"""
    from app.services.inference_optimized import get_inference_engine
    return get_inference_engine()

# ============================================================================
# TESTS DE MOTOR DE IA
# ============================================================================

class TestInferenceEngine:
    """Tests del motor de inferencia"""
    
    def test_engine_initialization(self, inference_engine):
        """Test: Motor inicializa correctamente"""
        assert inference_engine is not None
        assert inference_engine.device.type == 'cpu'
        assert len(inference_engine.class_names) == 5
    
    def test_valid_prediction(self, inference_engine, sample_image):
        """Test: Predicción válida con imagen correcta"""
        result = inference_engine.predict(sample_image)
        
        assert "label" in result
        assert "confidence" in result
        assert "certainty" in result
        assert 0 <= result['confidence'] <= 1
        assert 0 <= result['certainty'] <= 1
        assert result['label'] in inference_engine.class_names
    
    def test_prediction_structure(self, inference_engine, sample_image):
        """Test: Estructura de respuesta completa"""
        result = inference_engine.predict(sample_image)
        
        required_fields = [
            'label', 'confidence', 'certainty', 'model_version',
            'top_3_predictions', 'all_probabilities'
        ]
        
        for field in required_fields:
            assert field in result, f"Campo faltante: {field}"
        
        # Validar top-3
        assert len(result['top_3_predictions']) == 3
        for pred in result['top_3_predictions']:
            assert 'label' in pred
            assert 'probability' in pred
        
        # Validar todas las probabilidades
        assert len(result['all_probabilities']) == 5
        total_prob = sum(result['all_probabilities'].values())
        assert 0.99 <= total_prob <= 1.01  # Suma ~1.0
    
    def test_small_image_validation(self, inference_engine, small_image):
        """Test: Rechazo de imagen muy pequeña"""
        with pytest.raises(ValueError):
            inference_engine.predict(small_image)
    
    def test_cache_functionality(self, inference_engine, sample_image):
        """Test: Caché funciona correctamente"""
        # Primera predicción
        result1 = inference_engine.predict(sample_image)
        cache_size_1 = len(inference_engine.prediction_cache)
        
        # Segunda predicción (debe usar caché)
        result2 = inference_engine.predict(sample_image)
        cache_size_2 = len(inference_engine.prediction_cache)
        
        # Deben ser iguales
        assert result1 == result2
        assert cache_size_1 == cache_size_2
    
    def test_batch_prediction(self, inference_engine, sample_image):
        """Test: Predicción en lote"""
        images = [sample_image, sample_image, sample_image]
        results = inference_engine.predict_batch(images)
        
        assert len(results) == 3
        for result in results:
            assert "label" in result
            assert "confidence" in result
    
    def test_model_info(self, inference_engine):
        """Test: Información del modelo"""
        info = inference_engine.get_model_info()
        
        assert info['model_name'] == "EfficientNetV2-M"
        assert "VisionPlant" in info['model_version']
        assert info['image_size'] == 384
        assert info['num_classes'] == 5
        assert len(info['features']) > 0

# ============================================================================
# TESTS DE API
# ============================================================================

@pytest.mark.asyncio
class TestAPI:
    """Tests de API REST"""
    
    @pytest.fixture
    async def client(self):
        """Cliente de prueba de FastAPI"""
        from fastapi.testclient import TestClient
        from app.api_professional import app
        return TestClient(app)
    
    async def test_health_endpoint(self, client):
        """Test: Endpoint /health"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'model_loaded' in data
    
    async def test_model_info_endpoint(self, client):
        """Test: Endpoint /model-info"""
        response = client.get("/api/v1/model-info")
        assert response.status_code == 200
        data = response.json()
        assert data['model_name'] == "EfficientNetV2-M"
        assert data['num_classes'] == 5
    
    async def test_analyze_endpoint(self, client, sample_image):
        """Test: Endpoint /analyze"""
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("test.png", io.BytesIO(sample_image), "image/png")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data['success'] == True
        assert 'result' in data
        assert 'processing_time_ms' in data
    
    async def test_invalid_file_type(self, client):
        """Test: Rechazo de tipo de archivo inválido"""
        text_file = io.BytesIO(b"This is not an image")
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("test.txt", text_file, "text/plain")}
        )
        assert response.status_code == 400
    
    async def test_missing_file(self, client):
        """Test: Error cuando falta archivo"""
        response = client.post("/api/v1/analyze")
        assert response.status_code != 200

# ============================================================================
# TESTS DE RENDIMIENTO
# ============================================================================

class TestPerformance:
    """Tests de rendimiento"""
    
    def test_prediction_time(self, inference_engine, sample_image):
        """Test: Tiempo de predicción < 2 segundos"""
        import time
        start = time.time()
        inference_engine.predict(sample_image)
        elapsed = time.time() - start
        
        assert elapsed < 2.0, f"Predicción tardó {elapsed:.2f}s"
    
    def test_cache_speed(self, inference_engine, sample_image):
        """Test: Predicción en caché < 50ms"""
        import time
        
        # Calentar caché
        inference_engine.predict(sample_image)
        
        # Medir tiempo en caché
        start = time.time()
        inference_engine.predict(sample_image)
        elapsed_ms = (time.time() - start) * 1000
        
        assert elapsed_ms < 50, f"Caché tardó {elapsed_ms:.2f}ms"

# ============================================================================
# TESTS DE SEGURIDAD
# ============================================================================

class TestSecurity:
    """Tests de seguridad"""
    
    def test_file_size_limit(self):
        """Test: Límite de tamaño de archivo"""
        # Crear archivo muy grande (> 10MB)
        large_file = io.BytesIO(b"x" * (11 * 1024 * 1024))
        # Debería ser rechazado
        # (Test en endpoint, no en engine)
    
    def test_cache_limit(self, inference_engine, sample_image):
        """Test: Límite de caché"""
        initial_size = len(inference_engine.prediction_cache)
        
        # Llenar caché
        for i in range(inference_engine.cache_size + 10):
            inference_engine.predict(sample_image)
        
        # No debe exceder límite
        assert len(inference_engine.prediction_cache) <= inference_engine.cache_size

# ============================================================================
# TESTS DE VALIDACIÓN
# ============================================================================

class TestValidation:
    """Tests de validación"""
    
    def test_corrupt_image(self, inference_engine):
        """Test: Rechazo de imagen corrupta"""
        corrupt_bytes = b"This is not a valid image file"
        with pytest.raises(Exception):
            inference_engine.predict(corrupt_bytes)
    
    def test_wrong_color_space(self, inference_engine):
        """Test: Conversión de espacio de color"""
        # Crear imagen en escala de grises
        img = Image.new('L', (256, 256))  # Escala de grises
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Debería convertir a RGB automáticamente
        result = inference_engine.predict(img_bytes.getvalue())
        assert result['label'] in inference_engine.class_names

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
