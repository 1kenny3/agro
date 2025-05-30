import torch
import torch.nn as nn
import timm
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import cv2
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification
import warnings
warnings.filterwarnings('ignore')

from ..config.settings import settings

class AdvancedCropClassifier(nn.Module):
    """Продвинутая модель для классификации сельскохозяйственных культур с современными архитектурами"""
    
    def __init__(self, num_classes: int = 3, model_name: str = "swin_base_patch4_window7_224", pretrained: bool = True):
        super(AdvancedCropClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Загружаем предобученную модель
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # Адаптируем классификатор в зависимости от архитектуры
        if "swin" in model_name:
            # Для Swin Transformer моделей
            self.feature_size = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(
                nn.LayerNorm(self.feature_size),
                nn.Dropout(0.3),
                nn.Linear(self.feature_size, 512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
        elif "vit" in model_name:
            # Для Vision Transformer моделей
            self.feature_size = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(
                nn.LayerNorm(self.feature_size),
                nn.Dropout(0.2),
                nn.Linear(self.feature_size, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_classes)
            )
        elif "efficientnetv2" in model_name:
            # Для EfficientNetV2 моделей
            self.feature_size = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(self.feature_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
        elif "convnext" in model_name:
            # Для ConvNeXt моделей
            if hasattr(self.backbone.head, 'fc'):
                self.feature_size = self.backbone.head.fc.in_features
                self.backbone.head.fc = nn.Sequential(
                    nn.LayerNorm(self.feature_size),
                    nn.Dropout(0.3),
                    nn.Linear(self.feature_size, num_classes)
                )
            else:
                self.feature_size = self.backbone.head.in_features
                self.backbone.head = nn.Sequential(
                    nn.LayerNorm(self.feature_size),
                    nn.Dropout(0.3),
                    nn.Linear(self.feature_size, num_classes)
                )
        else:
            # Для других моделей (ResNet, EfficientNet и т.д.)
            if hasattr(self.backbone, 'classifier'):
                self.feature_size = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(self.feature_size, 512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(512, num_classes)
                )
            elif hasattr(self.backbone, 'fc'):
                self.feature_size = self.backbone.fc.in_features
                self.backbone.fc = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(self.feature_size, 512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(512, num_classes)
                )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход через модель"""
        return self.backbone(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Извлечение признаков без классификации"""
        if "swin" in self.model_name or "vit" in self.model_name:
            return self.backbone.forward_features(x)
        elif "convnext" in self.model_name:
            features = self.backbone.forward_features(x)
            if hasattr(self.backbone.head, 'global_pool'):
                return self.backbone.head.global_pool(features)
            return features.mean(dim=[-2, -1])  # Global average pooling
        else:
            features = self.backbone.forward_features(x)
            if hasattr(self.backbone, 'global_pool'):
                return self.backbone.global_pool(features)
            return features.mean(dim=[-2, -1])  # Global average pooling

class NextGenCropClassifier:
    """Классификатор нового поколения с современными нейронными сетями"""
    
    def __init__(self, device: str = None):
        self.device = device or settings.DEVICE
        self.classes = settings.CROP_CLASSES
        self.classes_ru = settings.CROP_CLASSES_RU
        
        # Создаем ансамбль из самых современных моделей
        self.models = self._create_next_gen_ensemble()
        
        # Веса для ансамбля (можно настроить на основе валидации)
        self.ensemble_weights = {
            "Swin-Transformer": 0.35,
            "Vision-Transformer": 0.30,
            "EfficientNetV2": 0.25,
            "ConvNeXt-V2": 0.10
        }
        
    def _create_next_gen_ensemble(self) -> List[Tuple[str, AdvancedCropClassifier]]:
        """Создаем ансамбль из самых современных моделей"""
        models = []
        
        # 1. Swin Transformer - лучший для сельскохозяйственных изображений
        print("🚀 Загружаю Swin Transformer - революционная архитектура для анализа растений...")
        try:
            swin_model = AdvancedCropClassifier(
                num_classes=len(self.classes),
                model_name="swin_base_patch4_window7_224",
                pretrained=True
            )
            swin_model.to(self.device)
            swin_model.eval()
            models.append(("Swin-Transformer", swin_model))
            print("✅ Swin Transformer успешно загружен! Готов к анализу структур растений")
        except Exception as e:
            print(f"⚠️ Swin Transformer не загрузился: {e}")
            # Fallback на меньшую версию
            try:
                swin_tiny = AdvancedCropClassifier(
                    num_classes=len(self.classes),
                    model_name="swin_tiny_patch4_window7_224",
                    pretrained=True
                )
                swin_tiny.to(self.device)
                swin_tiny.eval()
                models.append(("Swin-Transformer-Tiny", swin_tiny))
                print("✅ Swin Transformer Tiny загружен как резерв!")
            except Exception as e2:
                print(f"⚠️ Swin Transformer Tiny тоже не загрузился: {e2}")
        
        # 2. Vision Transformer - отличный для глобального понимания изображений
        print("🚀 Загружаю Vision Transformer - передовую архитектуру для компьютерного зрения...")
        try:
            vit_model = AdvancedCropClassifier(
                num_classes=len(self.classes),
                model_name="vit_base_patch16_224",
                pretrained=True
            )
            vit_model.to(self.device)
            vit_model.eval()
            models.append(("Vision-Transformer", vit_model))
            print("✅ Vision Transformer готов к работе! Превосходное глобальное понимание изображений")
        except Exception as e:
            print(f"⚠️ Vision Transformer не загрузился: {e}")
            # Fallback на меньшую версию
            try:
                vit_small = AdvancedCropClassifier(
                    num_classes=len(self.classes),
                    model_name="vit_small_patch16_224",
                    pretrained=True
                )
                vit_small.to(self.device)
                vit_small.eval()
                models.append(("Vision-Transformer-Small", vit_small))
                print("✅ Vision Transformer Small загружен!")
            except Exception as e2:
                print(f"⚠️ Vision Transformer Small тоже не загрузился: {e2}")
        
        # 3. EfficientNetV2 - улучшенная версия EfficientNet
        print("🚀 Загружаю EfficientNetV2 - оптимизированную архитектуру для точности и скорости...")
        try:
            efficientv2_model = AdvancedCropClassifier(
                num_classes=len(self.classes),
                model_name="efficientnetv2_s",
                pretrained=True
            )
            efficientv2_model.to(self.device)
            efficientv2_model.eval()
            models.append(("EfficientNetV2", efficientv2_model))
            print("✅ EfficientNetV2 готов! Оптимальный баланс точности и производительности")
        except Exception as e:
            print(f"⚠️ EfficientNetV2 не загрузился: {e}")
            # Fallback на обычный EfficientNet
            try:
                efficient_model = AdvancedCropClassifier(
                    num_classes=len(self.classes),
                    model_name="efficientnet_b3",
                    pretrained=True
                )
                efficient_model.to(self.device)
                efficient_model.eval()
                models.append(("EfficientNet-B3", efficient_model))
                print("✅ EfficientNet-B3 загружен как резерв!")
            except Exception as e2:
                print(f"⚠️ EfficientNet-B3 тоже не загрузился: {e2}")
        
        # 4. ConvNeXt V2 - современная CNN архитектура
        print("🚀 Загружаю ConvNeXt V2 - современную CNN архитектуру...")
        try:
            convnext_model = AdvancedCropClassifier(
                num_classes=len(self.classes),
                model_name="convnextv2_tiny",
                pretrained=True
            )
            convnext_model.to(self.device)
            convnext_model.eval()
            models.append(("ConvNeXt-V2", convnext_model))
            print("✅ ConvNeXt V2 готов! Современная CNN с улучшениями")
        except Exception as e:
            print(f"⚠️ ConvNeXt V2 не загрузился: {e}")
            # Fallback на обычный ConvNeXt
            try:
                convnext_model = AdvancedCropClassifier(
                    num_classes=len(self.classes),
                    model_name="convnext_tiny",
                    pretrained=True
                )
                convnext_model.to(self.device)
                convnext_model.eval()
                models.append(("ConvNeXt", convnext_model))
                print("✅ ConvNeXt загружен как резерв!")
            except Exception as e2:
                print(f"⚠️ ConvNeXt тоже не загрузился: {e2}")
        
        if not models:
            print("❌ Не удалось загрузить ни одну современную модель! Используем базовый ResNet50...")
            # Последний резерв - ResNet50
            try:
                resnet_model = AdvancedCropClassifier(
                    num_classes=len(self.classes),
                    model_name="resnet50",
                    pretrained=True
                )
                resnet_model.to(self.device)
                resnet_model.eval()
                models.append(("ResNet50-Fallback", resnet_model))
                print("✅ ResNet50 загружен как последний резерв")
            except Exception as e:
                print(f"❌ Критическая ошибка: даже ResNet50 не загрузился: {e}")
        
        print(f"\n🎯 Ансамбль готов! Загружено {len(models)} моделей:")
        for name, _ in models:
            print(f"   • {name}")
        
        return models
    
    def _get_advanced_transform(self) -> transforms.Compose:
        """Продвинутые трансформации для современных моделей"""
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _analyze_crop_morphology(self, image: Image.Image) -> Dict:
        """Продвинутый морфологический анализ растений"""
        img_array = np.array(image.convert('RGB'))
        
        # Анализ цветовых характеристик
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Анализ зеленого канала (хлорофилл)
        green_channel = img_array[:, :, 1]
        green_intensity = np.mean(green_channel) / 255.0
        green_std = np.std(green_channel) / 255.0
        
        # Анализ текстуры листьев
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Локальные бинарные паттерны для анализа текстуры
        def calculate_lbp_variance(image, radius=3):
            """Упрощенный анализ локальных паттернов"""
            kernel = np.ones((radius*2+1, radius*2+1), np.uint8)
            mean_filtered = cv2.filter2D(image.astype(np.float32), -1, kernel/(radius*2+1)**2)
            variance = cv2.filter2D((image.astype(np.float32) - mean_filtered)**2, -1, kernel/(radius*2+1)**2)
            return np.mean(variance)
        
        texture_variance = calculate_lbp_variance(gray)
        
        # Анализ структуры растения
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Направление градиентов
        angles = np.arctan2(sobely, sobelx)
        vertical_lines = np.sum(np.abs(angles) < np.pi/6) / angles.size
        horizontal_lines = np.sum(np.abs(angles - np.pi/2) < np.pi/6) / angles.size
        
        # Анализ формы листьев
        # Поиск контуров для анализа формы
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Анализ соотношения сторон листьев
        aspect_ratios = []
        leaf_areas = []
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Фильтруем мелкие контуры
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                aspect_ratios.append(aspect_ratio)
                leaf_areas.append(cv2.contourArea(contour))
        
        avg_aspect_ratio = np.mean(aspect_ratios) if aspect_ratios else 1.0
        total_leaf_area = sum(leaf_areas)
        
        # Специфический анализ для кукурузы
        corn_indicators = {
            "tassel_presence": self._detect_corn_tassel(hsv),
            "broad_leaves": avg_aspect_ratio > 2.0,
            "vertical_structure": vertical_lines > 0.3,
            "characteristic_green": 0.3 < green_intensity < 0.7
        }
        
        # Специфический анализ для пшеницы
        wheat_indicators = {
            "golden_color": self._detect_wheat_color(hsv),
            "thin_structure": avg_aspect_ratio > 3.0,
            "high_texture": texture_variance > 100,
            "uniform_pattern": green_std < 0.15
        }
        
        # Специфический анализ для ячменя
        barley_indicators = {
            "awns_presence": self._detect_barley_awns(gray),
            "compact_structure": avg_aspect_ratio < 2.5,
            "medium_texture": 50 < texture_variance < 150,
            "characteristic_color": self._detect_barley_color(hsv)
        }
        
        return {
            "green_intensity": green_intensity,
            "green_std": green_std,
            "texture_variance": texture_variance,
            "vertical_lines": vertical_lines,
            "horizontal_lines": horizontal_lines,
            "avg_aspect_ratio": avg_aspect_ratio,
            "total_leaf_area": total_leaf_area,
            "corn_score": sum(corn_indicators.values()) / len(corn_indicators),
            "wheat_score": sum(wheat_indicators.values()) / len(wheat_indicators),
            "barley_score": sum(barley_indicators.values()) / len(barley_indicators),
            "corn_indicators": corn_indicators,
            "wheat_indicators": wheat_indicators,
            "barley_indicators": barley_indicators
        }
    
    def _detect_corn_tassel(self, hsv_image: np.ndarray) -> bool:
        """Обнаружение метелок кукурузы"""
        upper_region = hsv_image[:hsv_image.shape[0]//3, :, :]
        # Цветовой диапазон для метелок (желто-коричневый)
        tassel_mask = cv2.inRange(upper_region, 
                                 np.array([10, 50, 50]), 
                                 np.array([30, 255, 255]))
        tassel_ratio = np.sum(tassel_mask > 0) / tassel_mask.size
        return tassel_ratio > 0.05
    
    def _detect_wheat_color(self, hsv_image: np.ndarray) -> bool:
        """Обнаружение характерного золотистого цвета пшеницы"""
        # Золотистый цвет пшеницы
        golden_mask = cv2.inRange(hsv_image,
                                 np.array([15, 50, 100]),
                                 np.array([35, 255, 255]))
        golden_ratio = np.sum(golden_mask > 0) / golden_mask.size
        return golden_ratio > 0.1
    
    def _detect_barley_awns(self, gray_image: np.ndarray) -> bool:
        """Обнаружение остей ячменя"""
        # Поиск тонких линейных структур (остей)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        opened = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
        awns_ratio = np.sum(opened > 0) / opened.size
        return awns_ratio > 0.05
    
    def _detect_barley_color(self, hsv_image: np.ndarray) -> bool:
        """Обнаружение характерного цвета ячменя"""
        # Светло-коричневый/бежевый цвет ячменя
        barley_mask = cv2.inRange(hsv_image,
                                 np.array([8, 30, 80]),
                                 np.array([25, 180, 220]))
        barley_ratio = np.sum(barley_mask > 0) / barley_mask.size
        return barley_ratio > 0.08
    
    def predict(self, image: Image.Image) -> Dict:
        """Продвинутое предсказание с использованием ансамбля современных моделей"""
        
        # Морфологический анализ
        morphology = self._analyze_crop_morphology(image)
        
        # Предобработка изображения
        transform = self._get_advanced_transform()
        tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Получаем предсказания от всех моделей
        ensemble_predictions = []
        model_results = {}
        model_confidences = {}
        
        for model_name, model in self.models:
            try:
                with torch.no_grad():
                    outputs = model(tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    probs_np = probabilities[0].cpu().numpy()
                    
                    model_results[model_name] = probs_np
                    model_confidences[model_name] = float(np.max(probs_np))
                    
                    # Применяем веса ансамбля
                    weight = self.ensemble_weights.get(model_name, 0.25)
                    weighted_probs = probs_np * weight
                    ensemble_predictions.append(weighted_probs)
                    
            except Exception as e:
                print(f"⚠️ Ошибка в модели {model_name}: {e}")
        
        if not ensemble_predictions:
            return self._create_error_result()
        
        # Взвешенное усреднение - исправляем ошибку с массивами
        try:
            # Убеждаемся, что все предсказания имеют одинаковую форму
            ensemble_predictions = [pred for pred in ensemble_predictions if pred is not None and len(pred) == len(self.classes)]
            
            if not ensemble_predictions:
                return self._create_error_result()
            
            # Преобразуем в numpy массив и усредняем
            ensemble_array = np.array(ensemble_predictions)
            final_probabilities = np.mean(ensemble_array, axis=0)
            
            # Нормализация
            final_probabilities = final_probabilities / np.sum(final_probabilities)
            
        except Exception as e:
            print(f"⚠️ Ошибка при усреднении ансамбля: {e}")
            # Fallback: используем результат первой модели
            if model_results:
                first_model_result = list(model_results.values())[0]
                final_probabilities = first_model_result / np.sum(first_model_result)
            else:
                return self._create_error_result()
        
        # Базовое предсказание
        predicted_class_idx = np.argmax(final_probabilities)
        base_predicted_class = self.classes[predicted_class_idx]
        base_confidence = float(final_probabilities[predicted_class_idx])
        
        # ИНТЕЛЛЕКТУАЛЬНАЯ КОРРЕКЦИЯ НА ОСНОВЕ МОРФОЛОГИИ
        morphology_scores = {
            "corn": morphology["corn_score"],
            "wheat": morphology["wheat_score"], 
            "barley": morphology["barley_score"]
        }
        
        # Находим лучший морфологический матч
        best_morphology_match = max(morphology_scores.keys(), key=lambda k: morphology_scores[k])
        morphology_confidence = morphology_scores[best_morphology_match]
        
        # Принятие финального решения
        analysis_notes = []
        
        # Если морфология сильно не согласна с нейросетью и имеет высокую уверенность
        if (best_morphology_match != base_predicted_class and 
            morphology_confidence > 0.7 and 
            base_confidence < 0.8):
            
            predicted_class = best_morphology_match
            predicted_class_ru = self.classes_ru[predicted_class]
            
            # Корректируем уверенность
            confidence = (base_confidence + morphology_confidence) / 2
            
            # Обновляем вероятности
            corrected_probs = final_probabilities.copy()
            morph_idx = self.classes.index(best_morphology_match)
            corrected_probs[morph_idx] = max(corrected_probs[morph_idx], confidence)
            corrected_probs = corrected_probs / np.sum(corrected_probs)
            
            final_probabilities = corrected_probs
            
            analysis_notes = [
                f"🔬 МОРФОЛОГИЧЕСКАЯ КОРРЕКЦИЯ: {predicted_class_ru}",
                f"🧠 Нейросети предсказали: {self.classes_ru[base_predicted_class]} ({base_confidence:.3f})",
                f"🔍 Морфология указывает на: {predicted_class_ru} ({morphology_confidence:.3f})",
                "✅ Применена интеллектуальная коррекция"
            ]
            
        else:
            # Стандартный результат с возможным усилением
            predicted_class = base_predicted_class
            predicted_class_ru = self.classes_ru[predicted_class]
            
            # Если морфология согласна, повышаем уверенность
            if best_morphology_match == base_predicted_class and morphology_confidence > 0.6:
                confidence = min(base_confidence + 0.1, 0.95)
                analysis_notes = [
                    f"✅ ПОДТВЕРЖДЕНО: {predicted_class_ru}",
                    f"🧠 Нейросети: {base_confidence:.3f}",
                    f"🔍 Морфология: {morphology_confidence:.3f}",
                    "🎯 Полное согласие всех методов анализа"
                ]
            else:
                confidence = base_confidence
                analysis_notes = [
                    f"🧠 Предсказание нейросетей: {predicted_class_ru}",
                    f"📊 Уверенность: {confidence:.3f}",
                    f"🔍 Морфологическая поддержка: {morphology_confidence:.3f}"
                ]
        
        # Создаем словари вероятностей
        all_probabilities = {}
        all_probabilities_ru = {}
        
        for i, class_name in enumerate(self.classes):
            prob = float(final_probabilities[i])
            all_probabilities[class_name] = prob
            all_probabilities_ru[self.classes_ru[class_name]] = prob
        
        # Определяем уровень уверенности
        if confidence >= 0.85:
            confidence_level = "Очень высокая"
            is_confident = True
        elif confidence >= 0.7:
            confidence_level = "Высокая"
            is_confident = True
        elif confidence >= 0.55:
            confidence_level = "Средняя"
            is_confident = True
        elif confidence >= 0.4:
            confidence_level = "Низкая"
            is_confident = False
        else:
            confidence_level = "Очень низкая"
            is_confident = False
        
        # Информация о моделях
        models_info = []
        for model_name, model_conf in model_confidences.items():
            models_info.append(f"{model_name}: {model_conf:.3f}")
        
        ensemble_info = f"🤖 Ансамбль из {len(self.models)} современных моделей"
        analysis_notes.insert(0, ensemble_info)
        analysis_notes.append(f"📈 Детали моделей: {', '.join(models_info)}")
        
        return {
            "predicted_class": predicted_class,
            "predicted_class_ru": predicted_class_ru,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "confidence_gap": float(np.max(final_probabilities) - np.partition(final_probabilities, -2)[-2]),
            "probabilities": all_probabilities,
            "probabilities_ru": all_probabilities_ru,
            "is_confident": is_confident,
            "analysis_notes": analysis_notes,
            "morphology_analysis": morphology,
            "model_results": model_results,
            "ensemble_method": "weighted_voting_with_morphology"
        }
    
    def _create_error_result(self) -> Dict:
        """Создает результат при ошибке"""
        return {
            "predicted_class": "uncertain",
            "predicted_class_ru": "неопределенный тип",
            "confidence": 0.0,
            "confidence_level": "Ошибка",
            "probabilities": {cls: 0.33 for cls in self.classes},
            "probabilities_ru": {self.classes_ru[cls]: 0.33 for cls in self.classes},
            "is_confident": False,
            "analysis_notes": ["❌ Ошибка в работе всех моделей ансамбля"],
            "morphology_analysis": {},
            "model_results": {},
            "ensemble_method": "error"
        } 