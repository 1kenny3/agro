import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from PIL import Image
import cv2
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import torchvision.transforms as transforms

from ..config.settings import settings

class YieldEstimationCNN(nn.Module):
    """CNN для извлечения признаков для прогнозирования урожайности"""
    
    def __init__(self, backbone_name: str = "resnet50", pretrained: bool = True):
        super(YieldEstimationCNN, self).__init__()
        
        import timm
        
        # Загружаем предобученную модель без последнего слоя
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=""
        )
        
        # Определяем размер признаков
        dummy_input = torch.randn(1, 3, settings.IMAGE_SIZE, settings.IMAGE_SIZE)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            self.feature_size = features.shape[1]
        
        # Адаптивный пулинг и регрессионная голова
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()  # Урожайность не может быть отрицательной
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход"""
        features = self.backbone(x)
        features = self.global_pool(features).flatten(1)
        yield_prediction = self.regressor(features)
        return yield_prediction
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Извлечение признаков без регрессии"""
        features = self.backbone(x)
        return self.global_pool(features).flatten(1)

class PlantCountEstimator:
    """Класс для подсчета растений на изображении"""
    
    def __init__(self):
        self.contour_area_threshold = 100
        self.morphology_kernel_size = 5
    
    def estimate_plant_density(self, image: Image.Image) -> Dict:
        """Оценка плотности растений на изображении"""
        # Конвертируем в OpenCV формат
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        # Преобразуем в HSV для лучшего выделения зеленых областей
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Диапазон зеленого цвета
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Создаем маску зеленых областей
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Морфологические операции для очистки маски
        kernel = np.ones((self.morphology_kernel_size, self.morphology_kernel_size), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        # Находим контуры
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Фильтруем контуры по площади
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.contour_area_threshold]
        
        # Вычисляем метрики
        total_green_area = sum(cv2.contourArea(c) for c in valid_contours)
        image_area = image.width * image.height
        green_coverage = total_green_area / image_area if image_area > 0 else 0
        
        # Оценка количества растений (упрощенная)
        plant_count = len(valid_contours)
        
        # Плотность растений (растений на единицу площади)
        density = plant_count / image_area * 10000  # нормализуем
        
        return {
            "plant_count": plant_count,
            "green_coverage": green_coverage,
            "density": density,
            "total_green_area": total_green_area,
            "image_area": image_area
        }

class YieldPredictor:
    """Класс для прогнозирования урожайности"""
    
    def __init__(self, cnn_model_path: str = None, ml_model_path: str = None, device: str = None):
        self.device = device or settings.DEVICE
        self.cnn_model = None
        self.ml_model = None
        self.scaler = StandardScaler()
        self.plant_counter = PlantCountEstimator()
        
        self.transform = transforms.Compose([
            transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        if cnn_model_path:
            self.load_cnn_model(cnn_model_path)
        
        if ml_model_path:
            self.load_ml_model(ml_model_path)
    
    def load_cnn_model(self, model_path: str) -> None:
        """Загрузка CNN модели"""
        self.cnn_model = YieldEstimationCNN(pretrained=False)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.cnn_model.load_state_dict(checkpoint)
        
        self.cnn_model.to(self.device)
        self.cnn_model.eval()
    
    def load_ml_model(self, model_path: str) -> None:
        """Загрузка ML модели"""
        import joblib
        self.ml_model = joblib.load(model_path)
    
    def extract_image_features(self, image: Image.Image) -> Dict:
        """Извлечение признаков из изображения"""
        features = {}
        
        # Признаки плотности растений (оставляем только основные - 3 признака)
        plant_features = self.plant_counter.estimate_plant_density(image)
        features["plant_count"] = int(plant_features["plant_count"])
        features["green_coverage"] = float(plant_features["green_coverage"])
        features["density"] = float(plant_features["density"])
        # Убираем total_green_area и image_area для экономии места
        
        # CNN признаки (оставляем 10)
        if self.cnn_model:
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                cnn_features = self.cnn_model.extract_features(tensor)
                # Берем первые 10 признаков CNN для упрощения
                for i in range(min(10, cnn_features.shape[1])):
                    features[f"cnn_feature_{i}"] = float(cnn_features[0][i].item())
        else:
            # Если CNN модель недоступна, заполняем нулями
            for i in range(10):
                features[f"cnn_feature_{i}"] = 0.0
        
        # Дополнительные визуальные признаки (5 признаков)
        img_array = np.array(image)
        
        # Средние значения цветовых каналов (3 признака)
        if len(img_array.shape) == 3:
            features["mean_red"] = float(np.mean(img_array[:, :, 0]))
            features["mean_green"] = float(np.mean(img_array[:, :, 1]))
            features["mean_blue"] = float(np.mean(img_array[:, :, 2]))
        else:
            mean_val = float(np.mean(img_array))
            features["mean_red"] = mean_val
            features["mean_green"] = mean_val
            features["mean_blue"] = mean_val
        
        # Контрастность и яркость (2 признака)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        features["brightness"] = float(np.mean(gray))
        features["contrast"] = float(np.std(gray))
        
        # Итого: 3 + 10 + 3 + 2 = 18 признаков, добавляем еще 2 для полноты
        features["aspect_ratio"] = float(image.width / image.height if image.height > 0 else 1.0)
        features["total_pixels"] = float(image.width * image.height)
        
        return features
    
    def predict_yield(self, image: Image.Image, external_features: Dict = None) -> Dict:
        """Прогнозирование урожайности"""
        # Извлекаем признаки из изображения
        image_features = self.extract_image_features(image)
        
        # Добавляем внешние признаки если есть
        if external_features:
            image_features.update(external_features)
        
        # Прогноз на основе CNN (если модель загружена)
        cnn_prediction = None
        if self.cnn_model:
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                cnn_prediction = self.cnn_model(tensor)[0].item()
        
        # Прогноз на основе ML модели (если модель загружена)
        ml_prediction = None
        if self.ml_model:
            # Определяем фиксированный порядок признаков для совместимости с обученной моделью
            feature_names = [
                "plant_count", "green_coverage", "density",
                "cnn_feature_0", "cnn_feature_1", "cnn_feature_2", "cnn_feature_3", "cnn_feature_4",
                "cnn_feature_5", "cnn_feature_6", "cnn_feature_7", "cnn_feature_8", "cnn_feature_9",
                "mean_red", "mean_green", "mean_blue", 
                "brightness", "contrast",
                "aspect_ratio", "total_pixels"
            ]
            
            # Подготавливаем признаки в правильном порядке
            feature_vector = np.array([image_features.get(name, 0.0) for name in feature_names]).reshape(1, -1)
            
            # Проверяем размерность
            if feature_vector.shape[1] != 20:
                raise ValueError(f"Ожидается 20 признаков, получено {feature_vector.shape[1]}")
            
            try:
                # Нормализуем признаки
                feature_vector_scaled = self.scaler.transform(feature_vector)
                
                # Делаем прогноз
                ml_prediction = self.ml_model.predict(feature_vector_scaled)[0]
            except Exception as e:
                print(f"Предупреждение: Ошибка ML прогноза: {e}")
                ml_prediction = None
        
        # Комбинированный прогноз
        predictions = [p for p in [cnn_prediction, ml_prediction] if p is not None]
        
        if predictions:
            # Простое усреднение предсказаний
            final_prediction = np.mean(predictions)
            prediction_std = np.std(predictions) if len(predictions) > 1 else 0.15 * final_prediction
        else:
            # Эвристический прогноз на основе плотности растений
            plant_density = image_features.get("density", 0)
            green_coverage = image_features.get("green_coverage", 0)
            
            # Простая эвристика (тонны/га)
            final_prediction = (plant_density * 0.1 + green_coverage * 5) * 2
            prediction_std = 0.15 * final_prediction
        
        # Уверенность в прогнозе
        confidence = max(0.5, min(0.95, 1.0 - (prediction_std / final_prediction) if final_prediction > 0 else 0.5))
        
        # Диапазон прогноза (±15%)
        lower_bound = max(0, final_prediction - 1.96 * prediction_std)
        upper_bound = final_prediction + 1.96 * prediction_std
        
        return {
            "predicted_yield_tons_per_ha": round(final_prediction, 2),
            "confidence": round(confidence, 3),
            "prediction_range": {
                "lower": round(lower_bound, 2),
                "upper": round(upper_bound, 2)
            },
            "individual_predictions": {
                "cnn_prediction": round(cnn_prediction, 2) if cnn_prediction else None,
                "ml_prediction": round(ml_prediction, 2) if ml_prediction else None
            },
            "features_used": image_features,
            "recommendations": self._generate_yield_recommendations(final_prediction, confidence, image_features)
        }
    
    def _generate_yield_recommendations(self, predicted_yield: float, confidence: float, features: Dict) -> List[str]:
        """Генерация рекомендаций на основе прогноза урожайности"""
        recommendations = []
        
        # Анализ прогнозируемой урожайности
        if predicted_yield < 2.0:
            recommendations.append("Низкая прогнозируемая урожайность. Рекомендуется анализ причин и корректирующие меры.")
        elif predicted_yield > 8.0:
            recommendations.append("Высокая прогнозируемая урожайность. Обеспечьте оптимальные условия для поддержания роста.")
        
        # Анализ уверенности
        if confidence < 0.7:
            recommendations.append("Низкая уверенность в прогнозе. Рекомендуется дополнительный мониторинг.")
        
        # Анализ плотности растений
        plant_density = features.get("density", 0)
        if plant_density < 0.1:
            recommendations.append("Низкая плотность растений. Возможно требуется пересев.")
        elif plant_density > 2.0:
            recommendations.append("Высокая плотность растений. Возможна необходимость прореживания.")
        
        # Анализ покрытия зеленью
        green_coverage = features.get("green_coverage", 0)
        if green_coverage < 0.3:
            recommendations.append("Низкое покрытие зеленью. Проверьте здоровье растений и условия роста.")
        
        if not recommendations:
            recommendations.append("Прогноз урожайности в пределах нормы. Продолжайте текущий уход.")
        
        return recommendations

def create_pretrained_yield_predictor() -> YieldPredictor:
    """Создание предобученной модели прогнозирования урожайности"""
    predictor = YieldPredictor()
    
    # Создаем и инициализируем CNN модель для демонстрации
    cnn_model = YieldEstimationCNN(pretrained=True)
    
    # Инициализируем веса регрессора
    with torch.no_grad():
        for layer in cnn_model.regressor:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
    
    predictor.cnn_model = cnn_model.to(predictor.device)
    predictor.cnn_model.eval()
    
    # Создаем простую XGBoost модель для демонстрации
    try:
        # Генерируем фиктивные данные для обучения базовой модели
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        X_dummy = np.random.randn(n_samples, n_features)
        y_dummy = (X_dummy[:, 0] * 2 + X_dummy[:, 1] * 1.5 + 
                  np.random.normal(0, 0.5, n_samples) + 5)  # Базовая урожайность ~5 т/га
        
        # Обучаем простую модель
        ml_model = xgb.XGBRegressor(n_estimators=50, random_state=42)
        ml_model.fit(X_dummy, y_dummy)
        
        predictor.ml_model = ml_model
        predictor.scaler.fit(X_dummy)
        
    except Exception as e:
        print(f"Предупреждение: Не удалось создать ML модель: {e}")
    
    return predictor