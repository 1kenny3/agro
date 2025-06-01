#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌽 ТЕСТИРОВАНИЕ РАСПОЗНАВАНИЯ КУКУРУЗЫ
=======================================================
Проверяем, правильно ли модель распознает изображение images.jpeg как кукурузу
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import timm

class TrainedCornClassifier:
    def __init__(self, model_path="data/models/enhanced_crop_classifier.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.class_to_idx = {'wheat': 0, 'corn': 1, 'barley': 2}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Трансформации
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.model = self.load_model()
    
    def load_model(self):
        """Загрузка обученной модели"""
        try:
            print(f"📥 Загрузка модели из {self.model_path}...")
            
            # Создаем модель с правильной архитектурой
            class EnhancedCropClassifier(nn.Module):
                def __init__(self, num_classes=3, model_name="efficientnet_b0", use_visual_features=True):
                    super().__init__()
                    
                    self.use_visual_features = use_visual_features
                    
                    # Используем timm для извлечения признаков
                    self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
                    
                    # Получаем размер выходных признаков
                    with torch.no_grad():
                        dummy_input = torch.randn(1, 3, 224, 224)
                        features = self.backbone(dummy_input)
                        self.feature_size = features.shape[1]
                    
                    print(f"🔧 Размер признаков backbone: {self.feature_size}")
                    
                    # Сеть для визуальных признаков
                    if use_visual_features:
                        self.visual_net = nn.Sequential(
                            nn.Linear(5, 32),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(32, 64),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(64, 32)
                        )
                        
                        # Комбинированный классификатор
                        self.classifier = nn.Sequential(
                            nn.Linear(self.feature_size + 32, 256),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(128, num_classes)
                        )
                    else:
                        self.classifier = nn.Sequential(
                            nn.Linear(self.feature_size, 256),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(256, num_classes)
                        )
                
                def forward(self, images, visual_features=None):
                    # Извлекаем признаки изображения
                    image_features = self.backbone(images)
                    
                    if self.use_visual_features and visual_features is not None:
                        # Обрабатываем визуальные признаки
                        visual_features = self.visual_net(visual_features)
                        
                        # Объединяем признаки
                        combined_features = torch.cat([image_features, visual_features], dim=1)
                    else:
                        combined_features = image_features
                    
                    # Классификация
                    output = self.classifier(combined_features)
                    return output
            
            model = EnhancedCropClassifier(num_classes=3, use_visual_features=True)
            
            # Загружаем веса
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            model.to(self.device)
            model.eval()
            
            best_accuracy = checkpoint.get('best_accuracy', 0)
            print(f"✅ Модель загружена! Лучшая точность при обучении: {best_accuracy:.2f}%")
            
            return model
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_visual_features(self, image_path):
        """Извлечение визуальных признаков"""
        try:
            # Загружаем изображение
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            
            h, w = image.shape[:2]
            
            # Конвертируем в RGB и HSV
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 1. Соотношение зеленого цвета
            green_mask = cv2.inRange(image_hsv, (35, 40, 40), (85, 255, 255))
            green_ratio = np.sum(green_mask > 0) / (h * w)
            
            # 2. Соотношение желтого цвета
            yellow_mask = cv2.inRange(image_hsv, (15, 40, 40), (35, 255, 255))
            yellow_ratio = np.sum(yellow_mask > 0) / (h * w)
            
            # 3. Вертикальные линии (Sobel фильтр)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            vertical_lines = np.mean(np.abs(sobel_y)) / (np.mean(np.abs(sobel_x)) + 1e-8)
            vertical_lines = min(vertical_lines, 2.0) / 2.0  # Нормализуем
            
            # 4. Плотность краев
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (h * w)
            
            # 5. Соотношение сторон
            aspect_ratio = w / h
            
            features = torch.tensor([
                green_ratio,
                yellow_ratio, 
                vertical_lines,
                edge_density,
                aspect_ratio
            ], dtype=torch.float32)
            
            return features
            
        except Exception as e:
            print(f"❌ Ошибка извлечения признаков: {e}")
            return torch.zeros(5, dtype=torch.float32)
    
    def predict(self, image_path):
        """Предсказание для изображения"""
        if self.model is None:
            return None
        
        try:
            # Загружаем и обрабатываем изображение
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Извлекаем визуальные признаки
            visual_features = self.extract_visual_features(image_path).unsqueeze(0).to(self.device)
            
            # Предсказание
            with torch.no_grad():
                outputs = self.model(image_tensor, visual_features)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()
            
            predicted_class = self.idx_to_class[predicted_idx]
            
            # Создаем словарь вероятностей
            all_probabilities = {}
            for idx, class_name in self.idx_to_class.items():
                all_probabilities[class_name] = probabilities[0][idx].item()
            
            class_names_ru = {'wheat': 'пшеница', 'corn': 'кукуруза', 'barley': 'ячмень'}
            
            return {
                'predicted_class': predicted_class,
                'predicted_class_ru': class_names_ru.get(predicted_class, predicted_class),
                'confidence': confidence,
                'probabilities': all_probabilities,
                'visual_features': {
                    'green_ratio': visual_features[0][0].item(),
                    'yellow_ratio': visual_features[0][1].item(),
                    'vertical_lines': visual_features[0][2].item(),
                    'edge_density': visual_features[0][3].item(),
                    'aspect_ratio': visual_features[0][4].item()
                }
            }
            
        except Exception as e:
            print(f"❌ Ошибка предсказания: {e}")
            return None

def test_corn_image():
    """Тестирование изображения images.jpeg"""
    print("🌽 ТЕСТИРОВАНИЕ РАСПОЗНАВАНИЯ КУКУРУЗЫ")
    print("=" * 50)
    
    image_path = "photo/images.jpeg"
    
    if not Path(image_path).exists():
        print(f"❌ Файл {image_path} не найден!")
        return
    
    # Создаем классификатор
    classifier = TrainedCornClassifier()
    
    if classifier.model is None:
        print("❌ Модель не загружена!")
        return
    
    print(f"\n📸 Тестирование файла: {image_path}")
    print("🎯 Ожидаемый результат: кукуруза")
    
    # Получаем предсказание
    result = classifier.predict(image_path)
    
    if result is None:
        print("❌ Не удалось получить предсказание!")
        return
    
    # Выводим результаты
    print(f"\n📊 РЕЗУЛЬТАТ РАСПОЗНАВАНИЯ:")
    print(f"   🌾 Предсказанная культура: {result['predicted_class_ru']}")
    print(f"   📈 Уверенность: {result['confidence']:.1%}")
    
    # Проверяем правильность
    is_correct = result['predicted_class'] == 'corn'
    status = "✅ ПРАВИЛЬНО!" if is_correct else "❌ НЕПРАВИЛЬНО!"
    print(f"   {status}")
    
    print(f"\n🔍 ДЕТАЛЬНЫЕ ВЕРОЯТНОСТИ:")
    for class_name, prob in result['probabilities'].items():
        class_ru = {'wheat': 'пшеница', 'corn': 'кукуруза', 'barley': 'ячмень'}[class_name]
        percentage = prob * 100
        print(f"   {class_ru}: {percentage:.1f}%")
    
    print(f"\n📊 ВИЗУАЛЬНЫЕ ПРИЗНАКИ:")
    for feature, value in result['visual_features'].items():
        print(f"   {feature}: {value:.3f}")
    
    return is_correct

if __name__ == "__main__":
    test_corn_image() 