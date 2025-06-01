from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple, Any

class ProcessingInfo(BaseModel):
    """Информация о предобработке изображения"""
    original_size: Tuple[int, int]
    final_size: Tuple[int, int]
    warnings: List[str]

class ImageQualityMetrics(BaseModel):
    """Метрики качества изображения"""
    sharpness: float
    contrast: float
    brightness: float
    noise_level: float
    blur_level: float
    histogram_spread: float
    dynamic_range: int
    quality_score: float
    quality_level: str
    recommendations: List[str]
    post_enhancement_score: Optional[float] = None
    improvement: Optional[float] = None

class AdvancedProcessingInfo(BaseModel):
    """Расширенная информация о предобработке"""
    original_size: Tuple[int, int]
    final_size: Tuple[int, int]
    enhancement_applied: bool
    quality_metrics: ImageQualityMetrics
    multi_scale_available: bool = False
    attention_maps_available: bool = False

class ClassificationResult(BaseModel):
    """Результат классификации"""
    predicted_class: str
    predicted_class_ru: str
    confidence: float
    probabilities: Dict[str, float]
    probabilities_ru: Dict[str, float]

class QualityResult(BaseModel):
    """Результат оценки одного аспекта качества с улучшенной информацией"""
    predicted_class: str
    predicted_class_ru: str
    confidence: float
    health_confidence: Optional[float] = None  # Для болезней
    probabilities: Dict[str, float]
    probabilities_ru: Dict[str, float]

class PredictionRange(BaseModel):
    """Диапазон прогноза"""
    lower: float
    upper: float

class IndividualPredictions(BaseModel):
    """Индивидуальные предсказания разных моделей"""
    cnn_prediction: Optional[float]
    ml_prediction: Optional[float]

class ErrorResponse(BaseModel):
    """Схема ответа об ошибке"""
    success: bool = False
    error: str
    detail: Optional[str] = None

class ImageQualityResponse(BaseModel):
    """Ответ на запрос оценки качества изображения"""
    success: bool
    quality_metrics: ImageQualityMetrics
    enhancement_recommended: bool
    processing_time: float

class ImageEnhancementResponse(BaseModel):
    """Ответ на запрос улучшения изображения"""
    success: bool
    enhancement_applied: bool
    original_quality: ImageQualityMetrics
    enhanced_quality: Optional[ImageQualityMetrics] = None
    improvement_achieved: Optional[float] = None
    processing_time: float
    # Note: Enhanced image would be returned as base64 or file download

class AdvancedClassificationResponse(BaseModel):
    """Улучшенный ответ классификации с продвинутой обработкой"""
    success: bool
    predicted_class: str
    predicted_class_ru: str
    confidence: float
    confidence_level: Optional[str] = None
    confidence_gap: Optional[float] = None
    probabilities: Dict[str, float]
    probabilities_ru: Dict[str, float]
    is_confident: bool
    analysis_notes: Optional[List[str]] = None
    processing_info: AdvancedProcessingInfo

class CropClassificationResponse(BaseModel):
    """Ответ на запрос классификации культуры с улучшенным анализом"""
    success: bool
    predicted_class: str
    predicted_class_ru: str
    confidence: float
    confidence_level: Optional[str] = None  # Новое поле
    confidence_gap: Optional[float] = None  # Новое поле
    probabilities: Dict[str, float]
    probabilities_ru: Dict[str, float]
    is_confident: bool
    analysis_notes: Optional[List[str]] = None  # Новое поле
    processing_info: Dict[str, Any]  # Изменено на более гибкий тип
    corn_analysis: Optional[Dict[str, Any]] = None  # Диагностика кукурузы
    morphology_analysis: Optional[Dict[str, Any]] = None  # Морфологический анализ
    model_results: Optional[Dict[str, Any]] = None  # Результаты отдельных моделей

class QualityAssessmentResponse(BaseModel):
    """Ответ на запрос оценки качества с улучшенным анализом болезней"""
    success: bool
    quality: QualityResult
    disease: QualityResult
    maturity: QualityResult
    overall_score: float
    overall_quality: str
    is_healthy: bool
    health_status: Optional[str] = None  # Новое поле
    health_analysis: Optional[List[str]] = None  # Новое поле
    recommendations: List[str]
    processing_info: ProcessingInfo

class YieldPredictionResponse(BaseModel):
    """Ответ на запрос прогнозирования урожайности"""
    success: bool
    predicted_yield_tons_per_ha: float
    confidence: float
    prediction_range: PredictionRange
    individual_predictions: IndividualPredictions
    recommendations: List[str]
    processing_info: ProcessingInfo

class ComprehensiveAnalysisResponse(BaseModel):
    """Ответ на запрос комплексного анализа"""
    success: bool
    crop_classification: Dict[str, Any]
    quality_assessment: Dict[str, Any]
    yield_prediction: Dict[str, Any]
    comprehensive_recommendations: List[str]
    processing_info: ProcessingInfo 