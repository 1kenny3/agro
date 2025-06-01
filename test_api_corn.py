#!/usr/bin/env python3
"""
Тест API для классификации кукурузы
"""

import requests
import io
from PIL import Image
import numpy as np
import json

def create_corn_test_image():
    """Создание тестового изображения кукурузы"""
    # Создаем изображение с характеристиками кукурузы
    width, height = 224, 336  # Вертикальное соотношение
    test_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Зеленые области (листья кукурузы)
    test_image[50:height-20, :, 1] = 120  
    test_image[50:height-20, :, 0] = 40   
    test_image[50:height-20, :, 2] = 60   
    
    # Желто-коричневые области сверху (метелки)
    test_image[0:50, :, 0] = 180  
    test_image[0:50, :, 1] = 160  
    test_image[0:50, :, 2] = 80   
    
    # Вертикальные полосы (стебли)
    for x in range(0, width, 15):
        test_image[50:height-20, x:x+3, :] = [60, 140, 40]
    
    return Image.fromarray(test_image)

def test_api_classification():
    """Тест API классификации"""
    print("🌽 Тестирование API классификации кукурузы")
    print("="*50)
    
    # Создаем тестовое изображение
    image = create_corn_test_image()
    
    # Сохраняем изображение в BytesIO
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Отправляем запрос к API
    try:
        print("📡 Отправка запроса к API...")
        
        files = {'file': ('test_corn.png', img_byte_arr, 'image/png')}
        response = requests.post('http://localhost:8000/classify', files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API ответил успешно!")
            print(f"Предсказанный класс: {result.get('predicted_class')} ({result.get('predicted_class_ru')})")
            print(f"Уверенность: {result.get('confidence', 0):.3f}")
            print(f"Уровень уверенности: {result.get('confidence_level', 'Неизвестен')}")
            
            print("\nВероятности:")
            for cls, prob in result.get('probabilities_ru', {}).items():
                print(f"   {cls}: {prob:.3f}")
            
            print("\nЗаметки анализа:")
            for note in result.get('analysis_notes', []):
                print(f"   • {note}")
            
            # Информация о морфологическом анализе кукурузы
            if 'corn_analysis' in result:
                corn_analysis = result['corn_analysis']
                print(f"\n🔍 Морфологический анализ кукурузы:")
                print(f"   Счет кукурузы: {corn_analysis.get('corn_score', 0):.3f}")
                print(f"   Метелки: {corn_analysis.get('tassel_ratio', 0):.3f}")
                print(f"   Вертикальные структуры: {corn_analysis.get('vertical_lines', 0):.3f}")
                print(f"   Широкие листья: {corn_analysis.get('broad_structure', 0):.3f}")
                print(f"   Кукуруза вероятна: {corn_analysis.get('is_corn_likely', False)}")
            
            # Проверяем результат
            if result.get('predicted_class') == 'corn':
                print("\n🎉 УСПЕХ: API правильно классифицировал кукурузу!")
            else:
                print(f"\n❌ ОШИБКА: Ожидалась кукуруза, получено {result.get('predicted_class')}")
                
        else:
            print(f"❌ Ошибка API: {response.status_code}")
            print(f"Ответ: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Ошибка: Не удалось подключиться к API. Убедитесь, что сервер запущен на localhost:8000")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

def test_comprehensive_analysis():
    """Тест комплексного анализа"""
    print("\n" + "="*50)
    print("🔬 Тестирование комплексного анализа")
    print("="*50)
    
    # Создаем тестовое изображение
    image = create_corn_test_image()
    
    # Сохраняем изображение в BytesIO
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    try:
        print("📡 Отправка запроса к комплексному анализу...")
        
        files = {'file': ('test_corn.png', img_byte_arr, 'image/png')}
        response = requests.post('http://localhost:8000/analyze', files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Комплексный анализ завершен!")
            
            # Классификация культуры
            crop_result = result.get('crop_classification', {})
            print(f"Культура: {crop_result.get('predicted_class_ru', 'Неизвестно')}")
            print(f"Уверенность: {crop_result.get('confidence', 0):.3f}")
            
            # Оценка качества
            quality_result = result.get('quality_assessment', {})
            print(f"Общее качество: {quality_result.get('overall_quality', 'Неизвестно')}")
            
            # Рекомендации
            recommendations = result.get('comprehensive_recommendations', [])
            print("\nРекомендации:")
            for rec in recommendations[:3]:  # Показываем первые 3
                print(f"   • {rec}")
                
        else:
            print(f"❌ Ошибка комплексного анализа: {response.status_code}")
            print(f"Ответ: {response.text}")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    print("🚀 Тестирование API исправления классификации кукурузы")
    print("Цель: Проверить что API правильно классифицирует кукурузу")
    
    # Основной тест классификации
    test_api_classification()
    
    # Тест комплексного анализа
    test_comprehensive_analysis()
    
    print("\n🎉 Тестирование API завершено!") 