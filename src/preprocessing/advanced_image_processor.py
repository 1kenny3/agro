import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from typing import Tuple, List, Dict, Optional, Union
import torch
import torchvision.transforms as transforms
from skimage import exposure, filters, measure, morphology
from skimage.segmentation import slic, mark_boundaries
from scipy import ndimage
import matplotlib.pyplot as plt

from ..config.settings import settings


class AdvancedImageEnhancer:
    """Продвинутый класс для улучшения качества изображений"""
    
    def __init__(self):
        self.adaptive_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
    def enhance_for_recognition(self, image: Image.Image) -> Image.Image:
        """Комплексное улучшение изображения для лучшего распознавания"""
        img_array = np.array(image)
        
        # 1. Коррекция освещения
        img_array = self._correct_illumination(img_array)
        
        # 2. Адаптивное улучшение контраста
        img_array = self._adaptive_contrast_enhancement(img_array)
        
        # 3. Шумоподавление с сохранением деталей
        img_array = self._denoise_preserve_details(img_array)
        
        # 4. Улучшение цветов для растений
        img_array = self._enhance_vegetation_colors(img_array)
        
        # 5. Повышение резкости для мелких деталей
        img_array = self._smart_sharpening(img_array)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _correct_illumination(self, img_array: np.ndarray) -> np.ndarray:
        """Коррекция неравномерного освещения"""
        # Конвертируем в LAB для работы с яркостью
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Создаем модель фона для коррекции освещения
        background = cv2.morphologyEx(l_channel, cv2.MORPH_OPEN, 
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        background = cv2.GaussianBlur(background, (21, 21), 0)
        
        # Нормализуем освещение
        corrected_l = cv2.divide(l_channel, background, scale=255)
        lab[:, :, 0] = corrected_l
        
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def _adaptive_contrast_enhancement(self, img_array: np.ndarray) -> np.ndarray:
        """Адаптивное улучшение контраста с CLAHE"""
        # Работаем в YUV пространстве для лучшего результата
        yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
        
        # Применяем CLAHE к Y-каналу (яркость)
        yuv[:, :, 0] = self.adaptive_clahe.apply(yuv[:, :, 0])
        
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    
    def _denoise_preserve_details(self, img_array: np.ndarray) -> np.ndarray:
        """Шумоподавление с сохранением деталей"""
        # Используем Non-local Means для качественного шумоподавления
        denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        
        # Смешиваем с оригиналом для сохранения деталей
        alpha = 0.7  # Вес обработанного изображения
        result = cv2.addWeighted(denoised, alpha, img_array, 1-alpha, 0)
        
        return result
    
    def _enhance_vegetation_colors(self, img_array: np.ndarray) -> np.ndarray:
        """Улучшение цветов растительности"""
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Увеличиваем насыщенность зеленых оттенков
        mask_green = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([85, 255, 255]))
        
        # Создаем расширенную маску
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_DILATE, kernel)
        
        # Применяем улучшения только к зеленым областям
        enhanced_hsv = hsv.copy()
        enhanced_hsv[mask_green > 0, 1] = np.clip(enhanced_hsv[mask_green > 0, 1] * 1.2, 0, 255)  # Насыщенность
        enhanced_hsv[mask_green > 0, 2] = np.clip(enhanced_hsv[mask_green > 0, 2] * 1.1, 0, 255)  # Яркость
        
        return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)
    
    def _smart_sharpening(self, img_array: np.ndarray) -> np.ndarray:
        """Умное повышение резкости"""
        # Создаем kernel для unsharp masking
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        
        # Применяем unsharp masking
        sharpened = cv2.filter2D(img_array, -1, kernel)
        
        # Смешиваем с оригиналом для контроля силы эффекта
        alpha = 0.3
        result = cv2.addWeighted(img_array, 1-alpha, sharpened, alpha, 0)
        
        return np.clip(result, 0, 255)


class QualityAwarePreprocessor:
    """Препроцессор с учетом качества изображения"""
    
    def __init__(self, target_size: int = None):
        self.target_size = target_size or settings.IMAGE_SIZE
        self.enhancer = AdvancedImageEnhancer()
        
    def assess_image_quality(self, image: Image.Image) -> Dict:
        """Оценка качества изображения"""
        img_array = np.array(image)
        
        # Конвертируем в grayscale для анализа
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 1. Оценка резкости (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Оценка контраста
        contrast = gray.std()
        
        # 3. Оценка яркости
        brightness = gray.mean()
        
        # 4. Оценка шума
        noise_level = self._estimate_noise(gray)
        
        # 5. Оценка размытия
        blur_level = self._estimate_blur(gray)
        
        # 6. Анализ гистограммы
        hist_analysis = self._analyze_histogram(gray)
        
        # Общая оценка качества
        quality_score = self._calculate_quality_score(
            sharpness, contrast, brightness, noise_level, blur_level, hist_analysis
        )
        
        return {
            "sharpness": float(sharpness),
            "contrast": float(contrast),
            "brightness": float(brightness),
            "noise_level": float(noise_level),
            "blur_level": float(blur_level),
            "histogram_spread": hist_analysis["spread"],
            "dynamic_range": hist_analysis["dynamic_range"],
            "quality_score": quality_score,
            "quality_level": self._get_quality_level(quality_score),
            "recommendations": self._get_enhancement_recommendations(
                sharpness, contrast, brightness, noise_level, blur_level
            )
        }
    
    def _estimate_noise(self, gray: np.ndarray) -> float:
        """Оценка уровня шума в изображении"""
        # Используем высокочастотные компоненты для оценки шума
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        convolved = cv2.filter2D(gray, cv2.CV_64F, kernel)
        noise = convolved.std()
        return noise
    
    def _estimate_blur(self, gray: np.ndarray) -> float:
        """Оценка уровня размытия"""
        # Используем градиентный метод
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Низкая величина градиента указывает на размытие
        blur_measure = 1.0 / (magnitude.mean() + 1e-6)
        return blur_measure
    
    def _analyze_histogram(self, gray: np.ndarray) -> Dict:
        """Анализ гистограммы изображения"""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # Разброс гистограммы
        spread = np.std(hist)
        
        # Динамический диапазон
        non_zero = np.where(hist > 0)[0]
        if len(non_zero) > 0:
            dynamic_range = non_zero[-1] - non_zero[0]
        else:
            dynamic_range = 0
        
        return {
            "spread": float(spread),
            "dynamic_range": int(dynamic_range)
        }
    
    def _calculate_quality_score(self, sharpness: float, contrast: float, 
                               brightness: float, noise: float, blur: float,
                               hist_analysis: Dict) -> float:
        """Вычисление общей оценки качества"""
        # Нормализуем метрики
        sharpness_norm = min(sharpness / 100.0, 1.0)
        contrast_norm = min(contrast / 50.0, 1.0)
        brightness_norm = 1.0 - abs(brightness - 128) / 128.0  # Оптимальная яркость ~128
        noise_norm = max(0, 1.0 - noise / 20.0)
        blur_norm = max(0, 1.0 - blur / 0.1)
        dynamic_range_norm = min(hist_analysis["dynamic_range"] / 200.0, 1.0)
        
        # Взвешенная сумма
        weights = {
            "sharpness": 0.25,
            "contrast": 0.20,
            "brightness": 0.15,
            "noise": 0.15,
            "blur": 0.15,
            "dynamic_range": 0.10
        }
        
        quality_score = (
            weights["sharpness"] * sharpness_norm +
            weights["contrast"] * contrast_norm +
            weights["brightness"] * brightness_norm +
            weights["noise"] * noise_norm +
            weights["blur"] * blur_norm +
            weights["dynamic_range"] * dynamic_range_norm
        )
        
        return min(max(quality_score, 0.0), 1.0)
    
    def _get_quality_level(self, score: float) -> str:
        """Определение уровня качества по баллам"""
        if score >= 0.8:
            return "отличное"
        elif score >= 0.6:
            return "хорошее"
        elif score >= 0.4:
            return "удовлетворительное"
        else:
            return "плохое"
    
    def _get_enhancement_recommendations(self, sharpness: float, contrast: float,
                                       brightness: float, noise: float, blur: float) -> List[str]:
        """Рекомендации по улучшению изображения"""
        recommendations = []
        
        if sharpness < 50:
            recommendations.append("Повысить резкость")
        if contrast < 30:
            recommendations.append("Увеличить контраст")
        if brightness < 80 or brightness > 180:
            recommendations.append("Скорректировать яркость")
        if noise > 15:
            recommendations.append("Применить шумоподавление")
        if blur > 0.05:
            recommendations.append("Устранить размытие")
            
        if not recommendations:
            recommendations.append("Изображение хорошего качества")
            
        return recommendations
    
    def adaptive_preprocess(self, image: Image.Image) -> Dict:
        """Адаптивная предобработка на основе качества изображения"""
        # Оценка качества
        quality_info = self.assess_image_quality(image)
        
        # Применяем улучшения на основе оценки
        enhanced_image = image
        
        if quality_info["quality_score"] < 0.7:
            # Изображение требует улучшения
            enhanced_image = self.enhancer.enhance_for_recognition(image)
            
            # Переоценка после улучшения
            post_quality = self.assess_image_quality(enhanced_image)
            quality_info["post_enhancement_score"] = post_quality["quality_score"]
            quality_info["improvement"] = post_quality["quality_score"] - quality_info["quality_score"]
        
        # Создание тензора для модели
        transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        tensor = transform(enhanced_image)
        
        return {
            "original_image": image,
            "enhanced_image": enhanced_image,
            "tensor": tensor,
            "quality_info": quality_info,
            "enhancement_applied": quality_info["quality_score"] < 0.7
        }


class AgriculturalAugmentation:
    """Специализированные аугментации для сельскохозяйственных изображений"""
    
    def __init__(self):
        self.base_augmentations = A.Compose([
            # Геометрические трансформации
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=15, p=0.6),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.6),
            ], p=0.8),
            
            # Цветовые трансформации
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.6),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.6),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.4),
            ], p=0.7),
            
            # Атмосферные условия
            A.OneOf([
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.3),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, 
                               num_flare_circles_lower=1, num_flare_circles_upper=2, p=0.2),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, 
                             num_shadows_upper=2, shadow_dimension=5, p=0.3),
            ], p=0.4),
            
            # Шум и размытие
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=0.4),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.4),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=0.3),
            ], p=0.3),
            
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.2),
            ], p=0.2),
            
            # Специфичные для сельского хозяйства
            A.RandomCrop(height=int(self.target_size * 0.9), width=int(self.target_size * 0.9), p=0.3),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, 
                          min_holes=5, min_height=4, min_width=4, 
                          fill_value=0, mask_fill_value=0, p=0.3),
        ])
        
        self.target_size = settings.IMAGE_SIZE
    
    def create_seasonal_variations(self, image: Image.Image) -> List[Image.Image]:
        """Создание сезонных вариаций изображения"""
        img_array = np.array(image)
        variations = []
        
        # Весенняя версия - более зеленая и яркая
        spring = self._apply_seasonal_filter(img_array, "spring")
        variations.append(Image.fromarray(spring))
        
        # Летняя версия - более насыщенная
        summer = self._apply_seasonal_filter(img_array, "summer")
        variations.append(Image.fromarray(summer))
        
        # Осенняя версия - более желтая
        autumn = self._apply_seasonal_filter(img_array, "autumn")
        variations.append(Image.fromarray(autumn))
        
        return variations
    
    def _apply_seasonal_filter(self, img_array: np.ndarray, season: str) -> np.ndarray:
        """Применение сезонного фильтра"""
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        if season == "spring":
            # Увеличиваем зеленый оттенок и яркость
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] - 5, 0, 179)  # Сдвиг в сторону зеленого
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.1, 0, 255)  # Больше насыщенности
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)  # Больше яркости
            
        elif season == "summer":
            # Увеличиваем насыщенность и немного уменьшаем яркость
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.95, 0, 255)
            
        elif season == "autumn":
            # Сдвигаем в сторону желто-оранжевых оттенков
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] + 10, 0, 179)  # Сдвиг к желтому
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.9, 0, 255)  # Меньше насыщенности
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    def augment_for_training(self, image: Image.Image, num_augmentations: int = 5) -> List[Image.Image]:
        """Создание аугментированных версий для обучения"""
        img_array = np.array(image)
        augmented_images = []
        
        for _ in range(num_augmentations):
            # Применяем случайные трансформации
            augmented = self.base_augmentations(image=img_array)["image"]
            augmented_images.append(Image.fromarray(augmented))
        
        return augmented_images


class FeatureEnhancedPreprocessor:
    """Препроцессор с улучшением признаков для нейронных сетей"""
    
    def __init__(self, target_size: int = None):
        self.target_size = target_size or settings.IMAGE_SIZE
        self.quality_processor = QualityAwarePreprocessor(target_size)
        self.augmenter = AgriculturalAugmentation()
        
    def create_multi_scale_features(self, image: Image.Image) -> Dict:
        """Создание мультимасштабных признаков"""
        # Разные масштабы для захвата различных деталей
        scales = [0.8, 1.0, 1.2]
        multi_scale_tensors = []
        
        for scale in scales:
            # Изменяем размер
            new_size = int(self.target_size * scale)
            scaled_image = image.resize((new_size, new_size), Image.Resampling.LANCZOS)
            
            # Обрезаем или дополняем до целевого размера
            if scale != 1.0:
                if scale > 1.0:
                    # Центральная обрезка
                    left = (new_size - self.target_size) // 2
                    top = (new_size - self.target_size) // 2
                    scaled_image = scaled_image.crop((
                        left, top, 
                        left + self.target_size, 
                        top + self.target_size
                    ))
                else:
                    # Добавляем отступы
                    new_img = Image.new('RGB', (self.target_size, self.target_size), (128, 128, 128))
                    offset = (self.target_size - new_size) // 2
                    new_img.paste(scaled_image, (offset, offset))
                    scaled_image = new_img
            
            # Преобразуем в тензор
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            tensor = transform(scaled_image)
            multi_scale_tensors.append(tensor)
        
        return {
            "multi_scale_tensors": torch.stack(multi_scale_tensors),
            "scales": scales
        }
    
    def create_attention_maps(self, image: Image.Image) -> Dict:
        """Создание карт внимания для выделения важных областей"""
        img_array = np.array(image)
        
        # 1. Карта текстур (используя Gabor фильтры)
        texture_map = self._create_texture_map(img_array)
        
        # 2. Карта цветов (выделение растительности)
        color_map = self._create_vegetation_map(img_array)
        
        # 3. Карта краев
        edge_map = self._create_edge_map(img_array)
        
        # Объединяем карты
        combined_attention = (texture_map + color_map + edge_map) / 3.0
        
        # Нормализуем
        combined_attention = (combined_attention - combined_attention.min()) / \
                           (combined_attention.max() - combined_attention.min() + 1e-8)
        
        return {
            "texture_attention": texture_map,
            "vegetation_attention": color_map,
            "edge_attention": edge_map,
            "combined_attention": combined_attention
        }
    
    def _create_texture_map(self, img_array: np.ndarray) -> np.ndarray:
        """Создание карты текстур"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Gabor фильтры для различных ориентаций
        texture_responses = []
        for theta in [0, 45, 90, 135]:
            kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta), 
                                      2*np.pi*0.5, 0.5, 0, ktype=cv2.CV_32F)
            response = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            texture_responses.append(response)
        
        # Объединяем ответы
        texture_map = np.mean(texture_responses, axis=0)
        return texture_map / 255.0
    
    def _create_vegetation_map(self, img_array: np.ndarray) -> np.ndarray:
        """Создание карты растительности"""
        # Используем индекс NDVI (упрощенная версия)
        red = img_array[:, :, 0].astype(float)
        green = img_array[:, :, 1].astype(float)
        blue = img_array[:, :, 2].astype(float)
        
        # Псевдо-NDVI на основе RGB
        vegetation_index = (green - red) / (green + red + 1e-8)
        
        # Нормализация
        vegetation_map = np.clip(vegetation_index, 0, 1)
        
        return vegetation_map
    
    def _create_edge_map(self, img_array: np.ndarray) -> np.ndarray:
        """Создание карты краев"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Небольшое размытие для смягчения краев
        edge_map = cv2.GaussianBlur(edges, (5, 5), 1)
        
        return edge_map / 255.0
    
    def comprehensive_preprocess(self, image: Image.Image, 
                               include_augmentations: bool = False) -> Dict:
        """Комплексная предобработка изображения"""
        # Базовая адаптивная предобработка
        base_result = self.quality_processor.adaptive_preprocess(image)
        
        # Мультимасштабные признаки
        multi_scale_result = self.create_multi_scale_features(base_result["enhanced_image"])
        
        # Карты внимания
        attention_maps = self.create_attention_maps(base_result["enhanced_image"])
        
        result = {
            **base_result,
            **multi_scale_result,
            "attention_maps": attention_maps
        }
        
        # Аугментации (если требуется)
        if include_augmentations:
            augmented_images = self.augmenter.augment_for_training(
                base_result["enhanced_image"], num_augmentations=3
            )
            seasonal_variations = self.augmenter.create_seasonal_variations(
                base_result["enhanced_image"]
            )
            
            result.update({
                "augmented_images": augmented_images,
                "seasonal_variations": seasonal_variations
            })
        
        return result


# Функция для быстрого доступа к улучшенной предобработке
def enhance_image_for_neural_network(image: Image.Image, 
                                   target_size: int = None,
                                   include_multi_scale: bool = True,
                                   include_attention: bool = True) -> Dict:
    """
    Быстрая функция для улучшения изображения для нейронных сетей
    
    Args:
        image: Входное изображение
        target_size: Целевой размер (по умолчанию из настроек)
        include_multi_scale: Включать мультимасштабные признаки
        include_attention: Включать карты внимания
    
    Returns:
        Словарь с обработанными данными
    """
    processor = FeatureEnhancedPreprocessor(target_size)
    
    if include_multi_scale and include_attention:
        return processor.comprehensive_preprocess(image)
    else:
        return processor.quality_processor.adaptive_preprocess(image) 