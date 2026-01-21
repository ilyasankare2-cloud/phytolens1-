"""
VisionPlant - Package de reconocimiento de plantas con IA avanzada
"""

__version__ = "1.0.0"
__author__ = "VisionPlant Team"

# Importar servicios principales (de forma segura)
# Las importaciones reales se harán cuando se necesiten
__all__ = [
    'PredictionCache',
    'GradCAM',
    'AdaptiveConfidence',
    'ExplainabilityEngine',
    'VisionPlantEdgeModel',
    'EdgeOptimizer',
    'FineTuningEngine',
    'FineTuningPipeline',
    'CustomPlantDataset',
]

# Lazy loading para evitar dependencias de importación
def __getattr__(name):
    if name == 'PredictionCache':
        from app.services.cache_manager import PredictionCache
        return PredictionCache
    elif name == 'GradCAM':
        from app.services.explainability import GradCAM
        return GradCAM
    elif name == 'AdaptiveConfidence':
        from app.services.explainability import AdaptiveConfidence
        return AdaptiveConfidence
    elif name == 'ExplainabilityEngine':
        from app.services.explainability import ExplainabilityEngine
        return ExplainabilityEngine
    elif name == 'VisionPlantEdgeModel':
        from app.services.inference_edge import VisionPlantEdgeModel
        return VisionPlantEdgeModel
    elif name == 'EdgeOptimizer':
        from app.services.inference_edge import EdgeOptimizer
        return EdgeOptimizer
    elif name == 'FineTuningEngine':
        from app.services.fine_tuning import FineTuningEngine
        return FineTuningEngine
    elif name == 'FineTuningPipeline':
        from app.services.fine_tuning import FineTuningPipeline
        return FineTuningPipeline
    elif name == 'CustomPlantDataset':
        from app.services.fine_tuning import CustomPlantDataset
        return CustomPlantDataset
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

