#!/usr/bin/env python3
"""
Тестирование обученной модели на фотографиях из папки photo/
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import numpy as np
from pathlib import Path
import cv2

from train_improved_model import EnhancedCropClassifier

class TrainedModelPredictor:
    """Предсказание с использованием обученной модели"""
    
    def __init__(self, model_path="data/models/enhanced_crop_classifier.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        self.class_to_idx = None
        self.idx_to_class = None
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.load_model()
    
    def load_model(self):
        """Загрузка обученной модели"""
        if not Path(self.model_path).exists():
            print(f"❌ Модель не найдена: {self.model_path}")
            print("Сначала обучите модель: python train_improved_model.py")
            return
        
        print(f"📥 Загрузка модели из {self.model_path}...")
        
        # Загружаем checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Создаем модель
        self.model = EnhancedCropClassifier(num_classes=3, use_visual_features=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Загружаем маппинг классов
        self.class_to_idx = checkpoint.get('class_to_idx', {'wheat': 0, 'corn': 1, 'barley': 2})
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        best_accuracy = checkpoint.get('val_acc', 0)
        print(f"✅ Модель загружена! Лучшая точность при обучении: {best_accuracy:.2f}%")
    
    def extract_visual_features(self, image_path):
        """Извлечение визуальных признаков"""
        try:
            image = Image.open(image_path)
            img_array = np.array(image.convert('RGB'))
            height, width = img_array.shape[:2]
            
            # Цветовой анализ
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Анализ зеленых областей
            green_mask = cv2.inRange(hsv, np.array([40, 40, 40]), np.array([80, 255, 255]))
            green_ratio = np.sum(green_mask > 0) / green_mask.size
            
            # Анализ желто-коричневых областей
            yellow_mask = cv2.inRange(hsv, np.array([10, 50, 50]), np.array([30, 255, 255]))
            yellow_ratio = np.sum(yellow_mask > 0) / yellow_mask.size
            
            # Анализ текстуры
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Поиск вертикальных структур
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            angles = np.arctan2(sobely, sobelx)
            vertical_lines = np.sum(np.abs(angles) < np.pi/6) / angles.size
            
            # Плотность краев
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Соотношение сторон
            aspect_ratio = height / width if width > 0 else 1.0
            
            return torch.tensor([
                green_ratio, yellow_ratio, vertical_lines, edge_density, aspect_ratio
            ], dtype=torch.float32)
            
        except Exception as e:
            print(f"⚠️ Ошибка извлечения признаков: {e}")
            return torch.zeros(5, dtype=torch.float32)
    
    def predict(self, image_path):
        """Предсказание для одного изображения"""
        if self.model is None:
            return None
        
        try:
            # Загружаем изображение
            image = Image.open(image_path).convert('RGB')
            
            # Преобразуем изображение
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
            print(f"❌ Ошибка предсказания для {image_path}: {e}")
            return None

def test_on_original_photos():
    """Тестирование на оригинальных фотографиях"""
    print("🧪 ТЕСТИРОВАНИЕ ОБУЧЕННОЙ МОДЕЛИ")
    print("=" * 50)
    
    # Создаем предиктор
    predictor = TrainedModelPredictor()
    
    if predictor.model is None:
        return
    
    # Загружаем результаты оригинального анализа
    with open("photo_analysis_results.json", 'r', encoding='utf-8') as f:
        original_results = json.load(f)
    
    print(f"\n📁 Тестирование на {len(original_results)} изображениях...")
    
    correct_predictions = 0
    total_predictions = 0
    
    results_comparison = []
    
    for result in original_results:
        file_path = result['file_info']['path']
        
        # Пропускаем файлы с ошибками
        if result.get('visual_features', {}).get('error'):
            continue
        
        print(f"\n📸 Тестирование: {Path(file_path).name}")
        
        # Предсказание обученной модели
        trained_prediction = predictor.predict(file_path)
        
        if trained_prediction is None:
            continue
        
        # Оригинальные результаты
        original_manual = result['manual_identification']
        original_ml = result['ml_prediction']
        
        # Сравнение
        is_correct_manual = trained_prediction['predicted_class'] == original_manual['class']
        is_correct_ml = trained_prediction['predicted_class'] == original_ml['class']
        
        total_predictions += 1
        if is_correct_manual:
            correct_predictions += 1
        
        print(f"   👨‍🌾 Ручная оценка: {original_manual['class_ru']}")
        print(f"   🤖 Старая модель: {original_ml['class_ru']} (уверенность: {original_ml['confidence']:.2f})")
        print(f"   🎯 Обученная модель: {trained_prediction['predicted_class_ru']} (уверенность: {trained_prediction['confidence']:.2f})")
        
        agreement_manual = "✅" if is_correct_manual else "❌"
        agreement_ml = "✅" if is_correct_ml else "❌"
        print(f"   {agreement_manual} Согласие с ручной оценкой")
        print(f"   {agreement_ml} Согласие со старой моделью")
        
        # Сохраняем для статистики
        results_comparison.append({
            'file': Path(file_path).name,
            'manual_class': original_manual['class'],
            'old_ml_class': original_ml['class'],
            'trained_class': trained_prediction['predicted_class'],
            'trained_confidence': trained_prediction['confidence'],
            'correct_vs_manual': is_correct_manual,
            'correct_vs_old_ml': is_correct_ml
        })
    
    # Итоговая статистика
    accuracy_vs_manual = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
    
    print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"   Общая точность vs ручная оценка: {accuracy_vs_manual:.1f}% ({correct_predictions}/{total_predictions})")
    
    # Статистика по классам
    class_stats = {}
    for result in results_comparison:
        manual_class = result['manual_class']
        if manual_class not in class_stats:
            class_stats[manual_class] = {'total': 0, 'correct': 0}
        
        class_stats[manual_class]['total'] += 1
        if result['correct_vs_manual']:
            class_stats[manual_class]['correct'] += 1
    
    print("\n   Точность по классам:")
    class_names_ru = {'wheat': 'пшеница', 'corn': 'кукуруза', 'barley': 'ячмень'}
    for class_name, stats in class_stats.items():
        accuracy = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        class_ru = class_names_ru.get(class_name, class_name)
        print(f"     {class_ru}: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
    
    # Сравнение со старой моделью
    old_model_agreements = sum(1 for r in results_comparison if r['correct_vs_old_ml'])
    old_model_accuracy = old_model_agreements / total_predictions * 100 if total_predictions > 0 else 0
    
    print(f"\n   Согласие с старой моделью: {old_model_accuracy:.1f}% ({old_model_agreements}/{total_predictions})")
    
    # Улучшения
    improvement = accuracy_vs_manual - 57.1  # 57.1% была согласованность в оригинальном анализе
    print(f"   Улучшение по сравнению с оригиналом: {improvement:+.1f}%")
    
    return results_comparison

def test_specific_image(image_path):
    """Тестирование конкретного изображения"""
    print(f"\n🔍 ДЕТАЛЬНЫЙ ТЕСТ ИЗОБРАЖЕНИЯ: {Path(image_path).name}")
    print("=" * 60)
    
    predictor = TrainedModelPredictor()
    
    if predictor.model is None:
        return
    
    result = predictor.predict(image_path)
    
    if result:
        print(f"📊 Результат предсказания:")
        print(f"   Культура: {result['predicted_class_ru']}")
        print(f"   Уверенность: {result['confidence']:.3f}")
        
        print(f"\n📈 Все вероятности:")
        class_names_ru = {'wheat': 'пшеница', 'corn': 'кукуруза', 'barley': 'ячмень'}
        for class_name, prob in result['probabilities'].items():
            class_ru = class_names_ru.get(class_name, class_name)
            print(f"   {class_ru}: {prob:.3f}")
        
        print(f"\n🔍 Визуальные признаки:")
        for feature, value in result['visual_features'].items():
            print(f"   {feature}: {value:.3f}")
    else:
        print("❌ Не удалось обработать изображение")

if __name__ == "__main__":
    print("🎯 ТЕСТИРОВАНИЕ ОБУЧЕННОЙ МОДЕЛИ НА ФОТОГРАФИЯХ")
    print("🌾 Проверка качества обучения на реальных данных")
    print("=" * 65)
    
    try:
        # Основное тестирование
        comparison_results = test_on_original_photos()
        
        # Детальный тест одного изображения
        photo_dir = Path("photo")
        image_files = [f for f in photo_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        if image_files:
            test_image = image_files[0]  # Берем первое изображение
            test_specific_image(test_image)
        
        print(f"\n🎉 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
        print(f"✅ Обученная модель протестирована на реальных фотографиях")
        
    except Exception as e:
        print(f"💥 Ошибка во время тестирования: {e}")
        import traceback
        traceback.print_exc() 