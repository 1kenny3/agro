#!/usr/bin/env python3
"""
Тест исправления классификации кукурузы
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from PIL import Image
import numpy as np
from src.models.crop_classifier import ImprovedCropClassifier
from src.config.settings import settings

def test_corn_classification():
    """Тест классификации кукурузы"""
    
    print("🌽 Тестирование улучшенной классификации кукурузы")
    print("="*60)
    
    # Инициализация улучшенного классификатора
    print("📋 Инициализация классификатора...")
    classifier = ImprovedCropClassifier()
    
    # Создаем тестовое изображение (имитация кукурузы)
    print("🖼️ Создание тестового изображения кукурузы...")
    
    # Создаем простое изображение с характеристиками кукурузы
    width, height = 224, 336  # Вертикальное соотношение сторон как у кукурузы
    test_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Добавляем зеленые области (листья)
    test_image[50:height-20, :, 1] = 120  # Зеленый канал
    test_image[50:height-20, :, 0] = 40   # Красный канал
    test_image[50:height-20, :, 2] = 60   # Синий канал
    
    # Добавляем желто-коричневые области сверху (метелки)
    test_image[0:50, :, 0] = 180  # Красный
    test_image[0:50, :, 1] = 160  # Зеленый
    test_image[0:50, :, 2] = 80   # Синий
    
    # Добавляем вертикальные полосы (стебли)
    for x in range(0, width, 15):
        test_image[50:height-20, x:x+3, :] = [60, 140, 40]
    
    # Конвертируем в PIL Image
    pil_image = Image.fromarray(test_image)
    
    print("📊 Анализ изображения...")
    
    # Тестируем анализ признаков кукурузы
    corn_features = classifier._analyze_corn_features(pil_image)
    
    print("\n🔍 Результаты анализа морфологических признаков:")
    for key, value in corn_features.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    # Тестируем полную классификацию
    print("\n🎯 Результаты классификации:")
    result = classifier.predict(pil_image)
    
    print(f"Предсказанный класс: {result['predicted_class']} ({result['predicted_class_ru']})")
    print(f"Уверенность: {result['confidence']:.3f}")
    print(f"Уровень уверенности: {result['confidence_level']}")
    print(f"Уверен ли классификатор: {result['is_confident']}")
    
    print("\nВероятности:")
    for cls, prob in result['probabilities_ru'].items():
        print(f"   {cls}: {prob:.3f}")
    
    print("\nЗаметки анализа:")
    for note in result['analysis_notes']:
        print(f"   • {note}")
    
    # Проверяем, что кукуруза правильно определена
    if result['predicted_class'] == 'corn':
        print("\n✅ УСПЕХ: Кукуруза правильно классифицирована!")
    else:
        print(f"\n❌ НЕУДАЧА: Ожидалась кукуруза, получено {result['predicted_class']}")
    
    return result

def test_with_existing_models():
    """Тест с существующими моделями"""
    print("\n" + "="*60)
    print("🧪 Тестирование с различными моделями")
    print("="*60)
    
    try:
        # Создаем простое изображение кукурузы для теста
        test_image = Image.new('RGB', (224, 300), color=(80, 120, 60))
        
        # Добавляем "метелки" сверху
        for y in range(0, 50):
            for x in range(0, 224):
                test_image.putpixel((x, y), (180, 160, 80))
        
        classifier = ImprovedCropClassifier()
        result = classifier.predict(test_image)
        
        print(f"Результат: {result['predicted_class_ru']}")
        print(f"Уверенность: {result['confidence']:.3f}")
        print(f"Морфологический анализ активен: {'corn_analysis' in result}")
        
        if 'corn_analysis' in result:
            corn_score = result['corn_analysis'].get('corn_score', 0)
            print(f"Счет кукурузы: {corn_score:.3f}")
        
    except Exception as e:
        print(f"Ошибка при тестировании: {e}")

if __name__ == "__main__":
    print("🚀 Запуск тестов исправления классификации кукурузы")
    print("Цель: Исправить проблему когда кукуруза определяется как ячмень")
    
    try:
        # Основной тест
        result = test_corn_classification()
        
        # Тест с моделями
        test_with_existing_models()
        
        print("\n🎉 Тестирование завершено!")
        
    except Exception as e:
        print(f"\n💥 Ошибка во время тестирования: {e}")
        import traceback
        traceback.print_exc() 