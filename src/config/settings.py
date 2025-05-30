import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List, Dict

class Settings(BaseSettings):
    """Основные настройки приложения"""
    
    # Общие настройки
    PROJECT_NAME: str = "Агропайплайн"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # API настройки
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    
    # Пути к данным
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = DATA_DIR / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # Настройки изображений
    IMAGE_SIZE: int = 224
    MAX_IMAGE_SIZE: int = 1024
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "bmp", "tiff"]
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Классы культур
    CROP_CLASSES: List[str] = ["wheat", "barley", "corn"]
    CROP_CLASSES_RU: Dict[str, str] = {
        "wheat": "пшеница",
        "barley": "ячмень", 
        "corn": "кукуруза"
    }
    
    # Классы качества
    QUALITY_CLASSES: List[str] = ["good", "fair", "poor"]
    QUALITY_CLASSES_RU: Dict[str, str] = {
        "good": "хорошее",
        "fair": "удовлетворительное",
        "poor": "плохое"
    }
    
    # Болезни
    DISEASE_CLASSES: List[str] = ["healthy", "rust", "powdery_mildew", "fusarium"]
    DISEASE_CLASSES_RU: Dict[str, str] = {
        "healthy": "здоровое",
        "rust": "ржавчина",
        "powdery_mildew": "мучнистая роса",
        "fusarium": "фузариоз"
    }
    
    # Стадии зрелости
    MATURITY_CLASSES: List[str] = ["immature", "mature", "overripe"]
    MATURITY_CLASSES_RU: Dict[str, str] = {
        "immature": "незрелый",
        "mature": "спелый",
        "overripe": "перезрелый"
    }
    
    # Модели
    CROP_MODEL_NAME: str = "efficientnet_b4"
    QUALITY_MODEL_NAME: str = "resnet50"
    DISEASE_MODEL_NAME: str = "yolov8n"
    
    # Поддерживаемые архитектуры моделей
    SUPPORTED_MODEL_ARCHITECTURES: Dict[str, Dict] = {
        "cnn": {
            "efficientnet_b4": {"input_size": 224},
            "resnet50": {"input_size": 224},
            "mobilenetv3_small_100": {"input_size": 224}
        },
        "transformer": {
            "vit_base_patch16_224": {"input_size": 224, "patch_size": 16},
            "vit_large_patch16_224": {"input_size": 224, "patch_size": 16}
        },
        "hybrid": {
            "convnext_tiny": {"input_size": 224},
            "convnext_small": {"input_size": 224}
        }
    }
    
    # Предпочтительная модель распознавания
    PREFERRED_MODEL: str = "convnext_tiny"
    FALLBACK_MODEL: str = "vit_base_patch16_224"
    
    # Пороги уверенности (снижены для большей честности моделей)
    CROP_CONFIDENCE_THRESHOLD: float = 0.6
    QUALITY_CONFIDENCE_THRESHOLD: float = 0.55
    DISEASE_CONFIDENCE_THRESHOLD: float = 0.5
    
    # Новые пороги для улучшенной логики
    HIGH_CONFIDENCE_THRESHOLD: float = 0.8
    MEDIUM_CONFIDENCE_THRESHOLD: float = 0.6
    LOW_CONFIDENCE_THRESHOLD: float = 0.4
    
    # Пороги для обнаружения болезней (более консервативные)
    HEALTHY_THRESHOLD: float = 0.7
    DISEASE_DETECTION_THRESHOLD: float = 0.6
    
    # Настройки качества прогнозов
    MIN_PREDICTION_CONFIDENCE: float = 0.3
    
    # Обучение моделей
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 50
    EARLY_STOPPING_PATIENCE: int = 10
    
    # Устройство вычислений
    DEVICE: str = "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"
    
    # Wandb настройки
    WANDB_PROJECT: str = "agro-pipeline"
    WANDB_ENTITY: str = os.getenv("WANDB_ENTITY", "")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Создаем глобальный экземпляр настроек
settings = Settings()

# Создаем необходимые директории
for directory in [
    settings.DATA_DIR,
    settings.RAW_DATA_DIR,
    settings.PROCESSED_DATA_DIR,
    settings.MODELS_DIR,
    settings.LOGS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True) 