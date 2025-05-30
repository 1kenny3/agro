from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import io
from typing import Dict, List, Optional
import logging
import traceback
import numpy as np

from ..config.settings import settings
from ..models.crop_classifier import (
    create_pretrained_crop_classifier, 
    create_enhanced_crop_classifier,
    SmartCropClassifier  # Новый умный классификатор
)
from ..models.quality_assessor import create_pretrained_quality_assessor
from ..models.yield_predictor import create_pretrained_yield_predictor
from ..preprocessing.image_processor import ImagePreprocessor
from ..preprocessing.advanced_image_processor import (
    QualityAwarePreprocessor,
    AdvancedImageEnhancer,
    enhance_image_for_neural_network
)
from .schemas import (
    CropClassificationResponse, 
    QualityAssessmentResponse, 
    YieldPredictionResponse,
    ComprehensiveAnalysisResponse,
    ErrorResponse
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API для анализа сельскохозяйственных культур",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене укажите конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные для моделей
crop_classifier = None
enhanced_crop_classifier = None  # Старая улучшенная модель
smart_crop_classifier = None     # Новый умный классификатор
quality_assessor = None
yield_predictor = None
image_processor = None
advanced_processor = None
image_enhancer = None

@app.on_event("startup")
async def startup_event():
    """Инициализация моделей при запуске"""
    global crop_classifier, enhanced_crop_classifier, smart_crop_classifier, quality_assessor, yield_predictor, image_processor, advanced_processor, image_enhancer
    
    logger.info("Загрузка моделей...")
    
    try:
        # Загрузка нового умного классификатора (высший приоритет)
        try:
            logger.info("🚀 Инициализация умного классификатора нового поколения...")
            smart_crop_classifier = SmartCropClassifier(prefer_advanced=True)
            logger.info("✅ Умный классификатор успешно загружен!")
            logger.info("🎯 Автоматический выбор лучшей архитектуры активирован")
            logger.info("🧠 Поддержка современных нейронных сетей включена")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось загрузить умный классификатор: {e}")
            smart_crop_classifier = None
        
        # Загрузка улучшенной модели классификации (средний приоритет)
        try:
            enhanced_crop_classifier = create_enhanced_crop_classifier()
            logger.info("🎯 Улучшенная система распознавания успешно загружена!")
            logger.info("✅ Ансамблевый подход активирован")
            logger.info("✅ Морфологический анализ включен")
            logger.info("✅ Специализация для кукурузы активна")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось загрузить улучшенную модель: {e}")
            logger.info("🔄 Используем стандартную модель...")
            enhanced_crop_classifier = None
        
        # Загрузка стандартной модели как резерв
        crop_classifier = create_pretrained_crop_classifier()
        quality_assessor = create_pretrained_quality_assessor()
        yield_predictor = create_pretrained_yield_predictor()
        image_processor = ImagePreprocessor()
        
        # Загрузка продвинутых процессоров
        advanced_processor = QualityAwarePreprocessor()
        image_enhancer = AdvancedImageEnhancer()
        
        logger.info("Модели успешно загружены")
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке моделей: {e}")
        raise

def get_models():
    """Dependency для получения моделей"""
    if not all([crop_classifier, quality_assessor, yield_predictor, image_processor]):
        raise HTTPException(status_code=503, detail="Модели не загружены")
    
    return {
        "crop_classifier": crop_classifier,
        "enhanced_crop_classifier": enhanced_crop_classifier,  # Старая улучшенная модель
        "smart_crop_classifier": smart_crop_classifier,        # Новый умный классификатор
        "quality_assessor": quality_assessor,
        "yield_predictor": yield_predictor,
        "image_processor": image_processor,
        "advanced_processor": advanced_processor,
        "image_enhancer": image_enhancer
    }

async def validate_and_process_image(file: UploadFile) -> Image.Image:
    """Валидация и предобработка загруженного изображения"""
    
    # Проверка размера файла
    contents = await file.read()
    if len(contents) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Файл слишком большой. Максимальный размер: {settings.MAX_FILE_SIZE / 1024 / 1024:.1f} MB"
        )
    
    # Попытка загрузки изображения
    try:
        image = Image.open(io.BytesIO(contents))
        # Проверяем, что это действительно изображение
        image.verify()
        # Переоткрываем файл, так как verify() портит объект
        image = Image.open(io.BytesIO(contents))
        return image
    except Exception as e:
        # Если не удалось открыть как изображение, проверяем content_type
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, 
                detail="Файл должен быть изображением"
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Не удалось открыть файл как изображение: {str(e)}"
            )

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Агропайплайн API",
        "version": settings.VERSION,
        "status": "активен",
        "endpoints": {
            "classify": "/classify - Классификация культуры",
            "quality": "/quality - Оценка качества",
            "yield": "/yield - Прогнозирование урожайности",
            "analyze": "/analyze - Комплексный анализ",
            "image_quality": "/image/quality - Оценка качества изображения",
            "image_enhance": "/image/enhance - Улучшение изображения",
            "classify_advanced": "/classify/advanced - Продвинутая классификация",
            "analyze_comprehensive": "/analyze/comprehensive - Комплексный анализ с улучшением"
        }
    }

@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    models_status = {
        "crop_classifier": crop_classifier is not None,
        "enhanced_crop_classifier": enhanced_crop_classifier is not None,
        "smart_crop_classifier": smart_crop_classifier is not None,  # Новый умный классификатор
        "quality_assessor": quality_assessor is not None,
        "yield_predictor": yield_predictor is not None,
        "image_processor": image_processor is not None,
        "advanced_processor": advanced_processor is not None,
        "image_enhancer": image_enhancer is not None
    }
    
    all_loaded = all([crop_classifier, quality_assessor, yield_predictor, image_processor])
    enhanced_available = enhanced_crop_classifier is not None
    smart_available = smart_crop_classifier is not None
    
    # Получаем информацию о типе умного классификатора
    smart_info = {}
    if smart_crop_classifier:
        try:
            smart_info = smart_crop_classifier.get_classifier_info()
        except:
            smart_info = {"type": "unknown"}
    
    return {
        "status": "healthy" if all_loaded else "degraded",
        "models": models_status,
        "device": settings.DEVICE,
        "enhanced_classification": enhanced_available,
        "smart_classification": smart_available,
        "smart_classifier_type": smart_info.get("type", "none"),
        "features": {
            "basic_processing": True,
            "enhanced_crop_classification": enhanced_available,
            "smart_crop_classification": smart_available,
            "next_gen_models": smart_info.get("type") == "NextGen",
            "swin_transformer": smart_info.get("type") == "NextGen",
            "vision_transformer": smart_info.get("type") == "NextGen",
            "efficientnetv2": smart_info.get("type") == "NextGen",
            "morphological_analysis": enhanced_available or smart_available,
            "corn_specialization": enhanced_available or smart_available,
            "ensemble_models": enhanced_available or smart_available,
            "advanced_processing": advanced_processor is not None,
            "image_enhancement": image_enhancer is not None,
            "quality_assessment": True,
            "multi_scale_features": True,
            "attention_maps": True
        }
    }

@app.post("/classify", response_model=CropClassificationResponse)
async def classify_crop(
    file: UploadFile = File(...),
    models: Dict = Depends(get_models)
):
    """Классификация сельскохозяйственной культуры с умным выбором лучшей модели"""
    try:
        # Валидация и предобработка изображения
        image = await validate_and_process_image(file)
        
        # Предобработка
        processed = models["image_processor"].preprocess_image(image)
        if not processed["is_valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Ошибка обработки изображения: {processed['errors']}"
            )
        
        # Выбираем лучший доступный классификатор (приоритет: умный -> улучшенный -> стандартный)
        if models.get("smart_crop_classifier"):
            logger.info("🚀 Используем УМНЫЙ классификатор с автоматическим выбором архитектуры")
            result = models["smart_crop_classifier"].predict(processed["processed_image"])
        elif models.get("enhanced_crop_classifier"):
            logger.info("🎯 Используем улучшенную модель с морфологическим анализом")
            result = models["enhanced_crop_classifier"].predict(processed["processed_image"])
        else:
            logger.info("⚙️ Используем стандартную модель")
            result = models["crop_classifier"].predict(processed["processed_image"])
        
        # Конвертируем numpy типы
        result = convert_numpy_types(result)
        
        return CropClassificationResponse(
            success=True,
            predicted_class=result["predicted_class"],
            predicted_class_ru=result["predicted_class_ru"],
            confidence=result["confidence"],
            confidence_level=result.get("confidence_level", "Средняя"),
            probabilities=result["probabilities"],
            probabilities_ru=result["probabilities_ru"],
            is_confident=result.get("is_confident", True),
            analysis_notes=result.get("analysis_notes", []),
            processing_info={
                "original_size": processed["original_size"],
                "final_size": processed["final_size"],
                "warnings": processed["warnings"],
                "is_valid": processed["is_valid"],
                "classifier_used": result.get("classifier_type", "unknown")  # Добавляем информацию о типе классификатора
            },
            corn_analysis=result.get("corn_analysis", {}),  # Диагностика кукурузы
            morphology_analysis=result.get("morphology_analysis", {}),  # Морфологический анализ
            model_results=result.get("model_results", {})  # Результаты отдельных моделей
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка классификации: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.post("/quality", response_model=QualityAssessmentResponse)
async def assess_quality(
    file: UploadFile = File(...),
    models: Dict = Depends(get_models)
):
    """Оценка качества сельскохозяйственной культуры"""
    try:
        # Валидация и предобработка изображения
        image = await validate_and_process_image(file)
        
        # Предобработка
        processed = models["image_processor"].preprocess_image(image)
        if not processed["is_valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Ошибка обработки изображения: {processed['errors']}"
            )
        
        # Оценка качества
        result = models["quality_assessor"].predict(processed["processed_image"])
        
        return QualityAssessmentResponse(
            success=True,
            quality=result["quality"],
            disease=result["disease"],
            maturity=result["maturity"],
            overall_score=result["overall_score"],
            overall_quality=result["overall_quality"],
            is_healthy=result["is_healthy"],
            health_status=result.get("health_status"),
            health_analysis=result.get("health_analysis"),
            recommendations=result["recommendations"],
            processing_info={
                "original_size": processed["original_size"],
                "final_size": processed["final_size"],
                "warnings": processed["warnings"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка оценки качества: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.post("/yield", response_model=YieldPredictionResponse)
async def predict_yield(
    file: UploadFile = File(...),
    models: Dict = Depends(get_models)
):
    """Прогнозирование урожайности"""
    try:
        # Валидация и предобработка изображения
        image = await validate_and_process_image(file)
        
        # Предобработка
        processed = models["image_processor"].preprocess_image(image)
        if not processed["is_valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Ошибка обработки изображения: {processed['errors']}"
            )
        
        # Прогнозирование урожайности
        result = models["yield_predictor"].predict_yield(processed["processed_image"])
        
        return YieldPredictionResponse(
            success=True,
            predicted_yield_tons_per_ha=result["predicted_yield_tons_per_ha"],
            confidence=result["confidence"],
            prediction_range=result["prediction_range"],
            individual_predictions=result["individual_predictions"],
            recommendations=result["recommendations"],
            processing_info={
                "original_size": processed["original_size"],
                "final_size": processed["final_size"],
                "warnings": processed["warnings"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка прогнозирования урожайности: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.post("/analyze", response_model=ComprehensiveAnalysisResponse)
async def comprehensive_analysis(
    file: UploadFile = File(...),
    models: Dict = Depends(get_models)
):
    """Комплексный анализ сельскохозяйственной культуры"""
    try:
        # Валидация и предобработка изображения
        image = await validate_and_process_image(file)
        
        # Предобработка
        processed = models["image_processor"].preprocess_image(image)
        if not processed["is_valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Ошибка обработки изображения: {processed['errors']}"
            )
        
        processed_image = processed["processed_image"]
        
        # Выполняем все анализы - используем ту же логику выбора модели, что и в /classify
        if models.get("enhanced_crop_classifier"):
            logger.info("🎯 Используем улучшенную модель для комплексного анализа")
            crop_result = models["enhanced_crop_classifier"].predict(processed_image)
        else:
            logger.info("⚙️ Используем стандартную модель для комплексного анализа")
            crop_result = models["crop_classifier"].predict(processed_image)
            
        quality_result = models["quality_assessor"].predict(processed_image)
        yield_result = models["yield_predictor"].predict_yield(processed_image)
        
        # Конвертируем numpy типы в обычные Python типы
        crop_result = convert_numpy_types(crop_result)
        quality_result = convert_numpy_types(quality_result)
        yield_result = convert_numpy_types(yield_result)
        
        # Генерируем общие рекомендации
        comprehensive_recommendations = _generate_comprehensive_recommendations(
            crop_result, quality_result, yield_result
        )
        
        return ComprehensiveAnalysisResponse(
            success=True,
            crop_classification=crop_result,
            quality_assessment=quality_result,
            yield_prediction=yield_result,
            comprehensive_recommendations=comprehensive_recommendations,
            processing_info={
                "original_size": processed["original_size"],
                "final_size": processed["final_size"],
                "warnings": processed["warnings"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка комплексного анализа: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

def _generate_comprehensive_recommendations(crop_result: Dict, quality_result: Dict, yield_result: Dict) -> List[str]:
    """Генерация комплексных рекомендаций"""
    recommendations = []
    
    # Анализ культуры
    crop_class = crop_result["predicted_class_ru"]
    if not crop_result["is_confident"]:
        recommendations.append(f"Низкая уверенность в определении культуры ({crop_class}). Рекомендуется дополнительная проверка.")
    
    # Анализ качества
    if not quality_result["is_healthy"]:
        disease = quality_result["disease"]["predicted_class_ru"]
        recommendations.append(f"Обнаружено заболевание: {disease}. Необходимо лечение.")
    
    overall_score = quality_result["overall_score"]
    if overall_score < 3.0:
        recommendations.append("Общее качество культуры низкое. Требуется комплексный подход к улучшению условий.")
    
    # Анализ урожайности
    predicted_yield = yield_result["predicted_yield_tons_per_ha"]
    yield_confidence = yield_result["confidence"]
    
    if predicted_yield < 3.0:
        recommendations.append("Прогнозируемая урожайность низкая. Рекомендуется анализ и улучшение агротехники.")
    elif predicted_yield > 7.0:
        recommendations.append("Высокая прогнозируемая урожайность. Поддерживайте текущие условия выращивания.")
    
    if yield_confidence < 0.7:
        recommendations.append("Низкая уверенность в прогнозе урожайности. Требуется дополнительный мониторинг.")
    
    # Комплексные рекомендации
    if overall_score > 3.5 and predicted_yield > 5.0 and quality_result["is_healthy"]:
        recommendations.append("Культура в отличном состоянии! Продолжайте текущий уход.")
    
    return recommendations

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Глобальный обработчик исключений"""
    logger.error(f"Необработанная ошибка: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Внутренняя ошибка сервера",
            "detail": str(exc) if settings.DEBUG else "Обратитесь к администратору"
        }
    )

def convert_numpy_types(obj):
    """Рекурсивно конвертирует numpy типы в обычные Python типы"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

@app.post("/image/quality")
async def assess_image_quality(
    file: UploadFile = File(...),
    models: Dict = Depends(get_models)
):
    """Оценка качества изображения"""
    import time
    
    try:
        start_time = time.time()
        
        # Валидация изображения
        image = await validate_and_process_image(file)
        
        # Оценка качества
        quality_info = models["advanced_processor"].assess_image_quality(image)
        
        # Определяем, рекомендуется ли улучшение
        enhancement_recommended = quality_info["quality_score"] < 0.7
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "quality_metrics": quality_info,
            "enhancement_recommended": enhancement_recommended,
            "processing_time": processing_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка оценки качества изображения: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.post("/image/enhance")
async def enhance_image(
    file: UploadFile = File(...),
    models: Dict = Depends(get_models)
):
    """Улучшение качества изображения"""
    import time
    import base64
    from io import BytesIO
    
    try:
        start_time = time.time()
        
        # Валидация изображения
        image = await validate_and_process_image(file)
        
        # Оценка оригинального качества
        original_quality = models["advanced_processor"].assess_image_quality(image)
        
        # Определяем необходимость улучшения
        enhancement_needed = original_quality["quality_score"] < 0.7
        
        enhanced_quality = None
        improvement_achieved = None
        enhanced_image_b64 = None
        
        if enhancement_needed:
            # Улучшаем изображение
            enhanced_image = models["image_enhancer"].enhance_for_recognition(image)
            
            # Оценка улучшенного качества
            enhanced_quality = models["advanced_processor"].assess_image_quality(enhanced_image)
            improvement_achieved = enhanced_quality["quality_score"] - original_quality["quality_score"]
            
            # Конвертируем улучшенное изображение в base64 для возврата
            buffered = BytesIO()
            enhanced_image.save(buffered, format="JPEG", quality=95)
            enhanced_image_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "enhancement_applied": enhancement_needed,
            "original_quality": original_quality,
            "enhanced_quality": enhanced_quality,
            "improvement_achieved": improvement_achieved,
            "processing_time": processing_time,
            "enhanced_image_base64": enhanced_image_b64
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка улучшения изображения: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.post("/classify/advanced")
async def advanced_classify_crop(
    file: UploadFile = File(...),
    use_multi_scale: bool = True,
    use_attention: bool = True,
    models: Dict = Depends(get_models)
):
    """Продвинутая классификация с улучшенной обработкой изображения"""
    try:
        # Валидация изображения
        image = await validate_and_process_image(file)
        
        # Продвинутая обработка
        result = enhance_image_for_neural_network(
            image,
            target_size=settings.IMAGE_SIZE,
            include_multi_scale=use_multi_scale,
            include_attention=use_attention
        )
        
        # Классификация с основным тензором
        classification_result = models["crop_classifier"].predict_tensor(result["tensor"])
        
        # Создаем расширенную информацию о обработке
        processing_info = {
            "original_size": result["original_image"].size,
            "final_size": result["enhanced_image"].size,
            "enhancement_applied": result["enhancement_applied"],
            "quality_metrics": result["quality_info"],
            "multi_scale_available": use_multi_scale,
            "attention_maps_available": use_attention
        }
        
        return {
            "success": True,
            "predicted_class": classification_result["predicted_class"],
            "predicted_class_ru": classification_result["predicted_class_ru"],
            "confidence": classification_result["confidence"],
            "confidence_level": classification_result.get("confidence_level"),
            "confidence_gap": classification_result.get("confidence_gap"),
            "probabilities": classification_result["probabilities"],
            "probabilities_ru": classification_result["probabilities_ru"],
            "is_confident": classification_result["is_confident"],
            "analysis_notes": classification_result.get("analysis_notes"),
            "processing_info": processing_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка продвинутой классификации: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.post("/analyze/comprehensive")
async def comprehensive_enhanced_analysis(
    file: UploadFile = File(...),
    use_enhancement: bool = True,
    models: Dict = Depends(get_models)
):
    """Комплексный анализ с продвинутой обработкой изображения"""
    try:
        # Валидация изображения
        image = await validate_and_process_image(file)
        
        if use_enhancement:
            # Используем продвинутую обработку
            enhanced_result = models["advanced_processor"].adaptive_preprocess(image)
            processed_image = enhanced_result["enhanced_image"]
            processing_info = {
                "original_size": enhanced_result["original_image"].size,
                "final_size": enhanced_result["enhanced_image"].size,
                "enhancement_applied": enhanced_result["enhancement_applied"],
                "quality_metrics": enhanced_result["quality_info"],
                "warnings": []
            }
        else:
            # Используем стандартную обработку
            processed = models["image_processor"].preprocess_image(image)
            if not processed["is_valid"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Ошибка обработки изображения: {processed['errors']}"
                )
            processed_image = processed["processed_image"]
            processing_info = {
                "original_size": processed["original_size"],
                "final_size": processed["final_size"],
                "enhancement_applied": False,
                "quality_metrics": None,
                "warnings": processed["warnings"]
            }
        
        # Выполняем все анализы
        crop_result = models["crop_classifier"].predict(processed_image)
        quality_result = models["quality_assessor"].predict(processed_image)
        yield_result = models["yield_predictor"].predict_yield(processed_image)
        
        # Конвертируем numpy типы
        crop_result = convert_numpy_types(crop_result)
        quality_result = convert_numpy_types(quality_result)
        yield_result = convert_numpy_types(yield_result)
        
        # Генерируем расширенные рекомендации
        comprehensive_recommendations = _generate_enhanced_recommendations(
            crop_result, quality_result, yield_result, processing_info
        )
        
        return {
            "success": True,
            "crop_classification": crop_result,
            "quality_assessment": quality_result,
            "yield_prediction": yield_result,
            "comprehensive_recommendations": comprehensive_recommendations,
            "processing_info": processing_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка комплексного анализа: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

def _generate_enhanced_recommendations(crop_result: Dict, quality_result: Dict, 
                                     yield_result: Dict, processing_info: Dict) -> List[str]:
    """Генерация расширенных рекомендаций с учетом качества изображения"""
    recommendations = []
    
    # Рекомендации по качеству изображения
    if processing_info.get("quality_metrics"):
        quality_metrics = processing_info["quality_metrics"]
        if quality_metrics["quality_score"] < 0.5:
            recommendations.append("Качество исходного изображения низкое. Для более точного анализа рекомендуется использовать изображения лучшего качества.")
        
        if quality_metrics.get("improvement") and quality_metrics["improvement"] > 0.1:
            recommendations.append(f"Качество изображения улучшено на {quality_metrics['improvement']:.1%}. Это повысило точность анализа.")
    
    # Базовые рекомендации
    base_recommendations = _generate_comprehensive_recommendations(crop_result, quality_result, yield_result)
    recommendations.extend(base_recommendations)
    
    return recommendations

@app.post("/classify/nextgen")
async def classify_crop_nextgen(
    file: UploadFile = File(...),
    models: Dict = Depends(get_models)
):
    """Классификация с использованием нейронных сетей нового поколения"""
    try:
        # Валидация и предобработка изображения
        image = await validate_and_process_image(file)
        
        # Проверяем доступность умного классификатора
        if not models.get("smart_crop_classifier"):
            raise HTTPException(
                status_code=503,
                detail="Умный классификатор нового поколения недоступен"
            )
        
        # Предобработка
        processed = models["image_processor"].preprocess_image(image)
        if not processed["is_valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Ошибка обработки изображения: {processed['errors']}"
            )
        
        logger.info("🚀 Запуск классификатора НОВОГО ПОКОЛЕНИЯ")
        
        # Получаем информацию о классификаторе
        classifier_info = models["smart_crop_classifier"].get_classifier_info()
        
        # Выполняем классификацию
        result = models["smart_crop_classifier"].predict(processed["processed_image"])
        
        # Конвертируем numpy типы
        result = convert_numpy_types(result)
        
        # Расширенный ответ с дополнительной информацией
        return {
            "success": True,
            "classifier_info": {
                "type": classifier_info.get("type", "unknown"),
                "description": classifier_info.get("description", ""),
                "available_advanced": classifier_info.get("available_advanced", False),
                "device": classifier_info.get("device", "unknown")
            },
            "prediction": {
                "predicted_class": result["predicted_class"],
                "predicted_class_ru": result["predicted_class_ru"],
                "confidence": result["confidence"],
                "confidence_level": result.get("confidence_level", "Средняя"),
                "probabilities": result["probabilities"],
                "probabilities_ru": result["probabilities_ru"],
                "is_confident": result.get("is_confident", True)
            },
            "analysis": {
                "notes": result.get("analysis_notes", []),
                "morphology": result.get("morphology_analysis", {}),
                "model_results": result.get("model_results", {}),
                "ensemble_method": result.get("ensemble_method", "unknown")
            },
            "processing_info": {
                "original_size": processed["original_size"],
                "final_size": processed["final_size"],
                "warnings": processed["warnings"],
                "is_valid": processed["is_valid"]
            },
            "features_used": {
                "modern_architectures": classifier_info.get("type") == "NextGen",
                "swin_transformer": classifier_info.get("type") == "NextGen",
                "vision_transformer": classifier_info.get("type") == "NextGen",
                "efficientnetv2": classifier_info.get("type") == "NextGen",
                "morphological_analysis": "morphology_analysis" in result,
                "ensemble_voting": "model_results" in result,
                "intelligent_correction": "КОРРЕКЦИЯ" in str(result.get("analysis_notes", []))
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка в классификаторе нового поколения: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.get("/models/info")
async def get_models_info(models: Dict = Depends(get_models)):
    """Получение подробной информации о загруженных моделях"""
    
    info = {
        "available_models": {
            "standard_classifier": models.get("crop_classifier") is not None,
            "enhanced_classifier": models.get("enhanced_crop_classifier") is not None,
            "smart_classifier": models.get("smart_crop_classifier") is not None,
            "quality_assessor": models.get("quality_assessor") is not None,
            "yield_predictor": models.get("yield_predictor") is not None
        },
        "smart_classifier_info": {},
        "capabilities": {
            "basic_classification": True,
            "enhanced_classification": False,
            "next_gen_classification": False,
            "morphological_analysis": False,
            "ensemble_models": False,
            "modern_architectures": False
        }
    }
    
    # Получаем детальную информацию об умном классификаторе
    if models.get("smart_crop_classifier"):
        try:
            smart_info = models["smart_crop_classifier"].get_classifier_info()
            info["smart_classifier_info"] = smart_info
            
            # Обновляем возможности
            info["capabilities"]["next_gen_classification"] = True
            info["capabilities"]["morphological_analysis"] = True
            info["capabilities"]["ensemble_models"] = True
            
            if smart_info.get("type") == "NextGen":
                info["capabilities"]["modern_architectures"] = True
                info["capabilities"]["swin_transformer"] = True
                info["capabilities"]["vision_transformer"] = True
                info["capabilities"]["efficientnetv2"] = True
                
        except Exception as e:
            logger.warning(f"Не удалось получить информацию об умном классификаторе: {e}")
    
    # Проверяем улучшенный классификатор
    if models.get("enhanced_crop_classifier"):
        info["capabilities"]["enhanced_classification"] = True
        if not info["capabilities"]["morphological_analysis"]:
            info["capabilities"]["morphological_analysis"] = True
        if not info["capabilities"]["ensemble_models"]:
            info["capabilities"]["ensemble_models"] = True
    
    return info

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=settings.API_HOST, 
        port=settings.API_PORT,
        reload=settings.DEBUG
    ) 