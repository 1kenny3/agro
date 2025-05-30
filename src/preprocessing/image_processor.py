import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from typing import Tuple, List, Dict, Optional
import torch
import torchvision.transforms as transforms

from ..config.settings import settings

class ImageValidator:
    """Класс для валидации изображений"""
    
    def __init__(self):
        self.allowed_extensions = settings.ALLOWED_EXTENSIONS
        self.max_file_size = settings.MAX_FILE_SIZE
        self.max_image_size = settings.MAX_IMAGE_SIZE
    
    def validate_image(self, image: Image.Image, filename: str = None) -> Dict:
        """Валидация изображения"""
        errors = []
        warnings = []
        
        # Проверка расширения файла
        if filename:
            extension = filename.split('.')[-1].lower()
            if extension not in self.allowed_extensions:
                errors.append(f"Неподдерживаемый формат файла: {extension}")
        
        # Проверка размера изображения
        width, height = image.size
        if width > self.max_image_size or height > self.max_image_size:
            warnings.append(f"Изображение слишком большое ({width}x{height}), будет уменьшено")
        
        # Проверка минимального размера
        if width < 100 or height < 100:
            errors.append(f"Изображение слишком маленькое ({width}x{height}), минимум 100x100")
        
        # Проверка цветового режима
        if image.mode not in ['RGB', 'RGBA', 'L']:
            warnings.append(f"Нестандартный цветовой режим: {image.mode}, будет конвертирован в RGB")
        
        # Проверка качества изображения
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            # Проверка на слишком темное изображение
            brightness = np.mean(img_array)
            if brightness < 30:
                warnings.append("Изображение очень темное, качество анализа может быть снижено")
            elif brightness > 240:
                warnings.append("Изображение очень яркое, качество анализа может быть снижено")
            
            # Проверка контрастности
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            contrast = np.std(gray)
            if contrast < 20:
                warnings.append("Низкий контраст изображения")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

class ImagePreprocessor:
    """Класс для предобработки изображений"""
    
    def __init__(self, target_size: int = None):
        self.target_size = target_size or settings.IMAGE_SIZE
        self.validator = ImageValidator()
        
        # Базовые трансформации
        self.base_transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Трансформации для аугментации
        self.augmentation_transform = A.Compose([
            A.Resize(self.target_size, self.target_size),
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=15, p=0.5),
            ], p=0.7),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.ColorJitter(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(p=0.3),
                A.GaussianBlur(blur_limit=(1, 3), p=0.3),
                A.MotionBlur(blur_limit=3, p=0.2),
            ], p=0.3),
        ])
    
    def preprocess_image(self, image: Image.Image, validate: bool = True) -> Dict:
        """Основная предобработка изображения"""
        result = {
            "original_size": image.size,
            "processed_image": None,
            "tensor": None,
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Валидация
        if validate:
            validation_result = self.validator.validate_image(image)
            result.update(validation_result)
            
            if not validation_result["is_valid"]:
                return result
        
        try:
            # Конвертация в RGB если необходимо
            if image.mode != 'RGB':
                image = image.convert('RGB')
                result["warnings"].append(f"Конвертировано в RGB")
            
            # Уменьшение размера если необходимо
            if max(image.size) > settings.MAX_IMAGE_SIZE:
                ratio = settings.MAX_IMAGE_SIZE / max(image.size)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                result["warnings"].append(f"Размер уменьшен до {new_size}")
            
            # Улучшение качества изображения
            enhanced_image = self._enhance_image_quality(image)
            
            # Создание тензора
            tensor = self.base_transform(enhanced_image)
            
            result.update({
                "processed_image": enhanced_image,
                "tensor": tensor,
                "final_size": enhanced_image.size
            })
            
        except Exception as e:
            result.update({
                "is_valid": False,
                "errors": [f"Ошибка при обработке: {str(e)}"]
            })
        
        return result
    
    def _enhance_image_quality(self, image: Image.Image) -> Image.Image:
        """Улучшение качества изображения"""
        # Автоматическая коррекция контраста
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        # Небольшое повышение резкости
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.05)
        
        # Автоматическая коррекция цвета
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.05)
        
        return image
    
    def augment_image(self, image: Image.Image) -> Image.Image:
        """Аугментация изображения"""
        img_array = np.array(image)
        
        # Применяем аугментации
        augmented = self.augmentation_transform(image=img_array)
        
        return Image.fromarray(augmented["image"])
    
    def create_training_batch(self, images: List[Image.Image], 
                            apply_augmentation: bool = True) -> torch.Tensor:
        """Создание батча для обучения"""
        tensors = []
        
        for image in images:
            if apply_augmentation:
                # Случайно применяем аугментацию
                if np.random.random() > 0.5:
                    image = self.augment_image(image)
            
            # Предобработка
            result = self.preprocess_image(image, validate=False)
            if result["is_valid"]:
                tensors.append(result["tensor"])
        
        if tensors:
            return torch.stack(tensors)
        else:
            raise ValueError("Не удалось обработать ни одно изображение")

class BackgroundRemover:
    """Класс для удаления фона с изображений"""
    
    def __init__(self):
        self.threshold_methods = {
            "green": self._green_threshold,
            "adaptive": self._adaptive_threshold,
            "kmeans": self._kmeans_segmentation
        }
    
    def remove_background(self, image: Image.Image, method: str = "green") -> Image.Image:
        """Удаление фона с изображения"""
        if method not in self.threshold_methods:
            raise ValueError(f"Неподдерживаемый метод: {method}")
        
        img_array = np.array(image)
        mask = self.threshold_methods[method](img_array)
        
        # Применяем маску
        result = img_array.copy()
        result[mask == 0] = [255, 255, 255]  # Белый фон
        
        return Image.fromarray(result)
    
    def _green_threshold(self, img_array: np.ndarray) -> np.ndarray:
        """Выделение зеленых областей (растений)"""
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Диапазон зеленого цвета
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Морфологические операции для очистки
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _adaptive_threshold(self, img_array: np.ndarray) -> np.ndarray:
        """Адаптивная пороговая обработка"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Адаптивная бинаризация
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Инвертируем маску (растения темнее фона)
        mask = 255 - mask
        
        return mask
    
    def _kmeans_segmentation(self, img_array: np.ndarray) -> np.ndarray:
        """K-means сегментация"""
        # Подготавливаем данные
        data = img_array.reshape((-1, 3))
        data = np.float32(data)
        
        # K-means кластеризация
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Определяем, какой кластер соответствует растениям
        centers = np.uint8(centers)
        
        # Предполагаем, что растения более зеленые
        green_scores = centers[:, 1] - (centers[:, 0] + centers[:, 2]) / 2
        plant_cluster = np.argmax(green_scores)
        
        # Создаем маску
        mask = (labels.flatten() == plant_cluster).astype(np.uint8) * 255
        mask = mask.reshape(img_array.shape[:2])
        
        return mask

class CropExtractor:
    """Класс для извлечения областей с культурами"""
    
    def __init__(self):
        self.background_remover = BackgroundRemover()
    
    def extract_crop_regions(self, image: Image.Image, 
                           min_area: int = 1000) -> List[Image.Image]:
        """Извлечение областей с культурами"""
        img_array = np.array(image)
        
        # Получаем маску растений
        mask = self.background_remover._green_threshold(img_array)
        
        # Находим контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Фильтруем контуры по площади
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # Извлекаем области
        crop_regions = []
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Добавляем отступы
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_array.shape[1] - x, w + 2 * padding)
            h = min(img_array.shape[0] - y, h + 2 * padding)
            
            # Извлекаем область
            crop_region = img_array[y:y+h, x:x+w]
            crop_regions.append(Image.fromarray(crop_region))
        
        return crop_regions
    
    def analyze_crop_distribution(self, image: Image.Image) -> Dict:
        """Анализ распределения культур на изображении"""
        regions = self.extract_crop_regions(image)
        
        total_area = image.width * image.height
        crop_area = sum(region.width * region.height for region in regions)
        
        return {
            "num_regions": len(regions),
            "total_crop_area": crop_area,
            "crop_coverage": crop_area / total_area if total_area > 0 else 0,
            "average_region_size": crop_area / len(regions) if regions else 0,
            "regions": regions
        } 