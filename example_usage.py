#!/usr/bin/env python3
"""
Пример использования Агропайплайн API
"""

import requests
import json
from pathlib import Path
from PIL import Image
import io

# Конфигурация
API_BASE_URL = "http://localhost:8000"

def test_api_health():
    """Проверка состояния API"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API сервер работает")
            health_data = response.json()
            print(f"📊 Статус: {health_data['status']}")
            print(f"💻 Устройство: {health_data['device']}")
            print(f"🧠 Модели: {health_data['models']}")
        else:
            print(f"❌ API недоступен: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Не удалось подключиться к API. Убедитесь, что сервер запущен.")
        return False
    
    return True

def create_sample_image():
    """Создание примера изображения для тестирования"""
    # Создаем простое зеленое изображение (имитация растения)
    img = Image.new('RGB', (400, 400), color='green')
    
    # Сохраняем в буфер
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='JPEG')
    img_buffer.seek(0)
    
    return img_buffer

def test_classification():
    """Тестирование классификации культур"""
    print("\n🌾 Тестирование классификации культур...")
    
    img_buffer = create_sample_image()
    
    files = {'file': ('test_crop.jpg', img_buffer, 'image/jpeg')}
    
    try:
        response = requests.post(f"{API_BASE_URL}/classify", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Классификация выполнена успешно")
            print(f"🌾 Культура: {result['predicted_class_ru']}")
            print(f"📊 Уверенность: {result['confidence']:.1%}")
            print(f"🎯 Высокая уверенность: {'Да' if result['is_confident'] else 'Нет'}")
        else:
            print(f"❌ Ошибка классификации: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"❌ Ошибка при тестировании классификации: {e}")

def test_quality_assessment():
    """Тестирование оценки качества"""
    print("\n🔍 Тестирование оценки качества...")
    
    img_buffer = create_sample_image()
    
    files = {'file': ('test_crop.jpg', img_buffer, 'image/jpeg')}
    
    try:
        response = requests.post(f"{API_BASE_URL}/quality", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Оценка качества выполнена успешно")
            print(f"⭐ Общая оценка: {result['overall_score']:.1f}/5.0 ({result['overall_quality']})")
            print(f"🏥 Здоровье: {'Здоровое' if result['is_healthy'] else 'Болезнь обнаружена'}")
            print(f"🦠 Болезни: {result['disease']['predicted_class_ru']}")
            print(f"🌱 Зрелость: {result['maturity']['predicted_class_ru']}")
            
            if result['recommendations']:
                print("📋 Рекомендации:")
                for i, rec in enumerate(result['recommendations'], 1):
                    print(f"   {i}. {rec}")
        else:
            print(f"❌ Ошибка оценки качества: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"❌ Ошибка при тестировании оценки качества: {e}")

def test_yield_prediction():
    """Тестирование прогнозирования урожайности"""
    print("\n📈 Тестирование прогнозирования урожайности...")
    
    img_buffer = create_sample_image()
    
    files = {'file': ('test_crop.jpg', img_buffer, 'image/jpeg')}
    
    try:
        response = requests.post(f"{API_BASE_URL}/yield", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Прогнозирование урожайности выполнено успешно")
            print(f"📊 Прогноз: {result['predicted_yield_tons_per_ha']:.2f} т/га")
            print(f"🎯 Уверенность: {result['confidence']:.1%}")
            
            pred_range = result['prediction_range']
            print(f"📈 Диапазон: {pred_range['lower']:.1f} - {pred_range['upper']:.1f} т/га")
            
            if result['recommendations']:
                print("📋 Рекомендации:")
                for i, rec in enumerate(result['recommendations'], 1):
                    print(f"   {i}. {rec}")
        else:
            print(f"❌ Ошибка прогнозирования урожайности: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"❌ Ошибка при тестировании прогнозирования урожайности: {e}")

def test_comprehensive_analysis():
    """Тестирование комплексного анализа"""
    print("\n🎯 Тестирование комплексного анализа...")
    
    img_buffer = create_sample_image()
    
    files = {'file': ('test_crop.jpg', img_buffer, 'image/jpeg')}
    
    try:
        response = requests.post(f"{API_BASE_URL}/analyze", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Комплексный анализ выполнен успешно")
            
            # Классификация
            crop_result = result['crop_classification']
            print(f"🌾 Культура: {crop_result['predicted_class_ru']} ({crop_result['confidence']:.1%})")
            
            # Качество
            quality_result = result['quality_assessment']
            print(f"⭐ Качество: {quality_result['overall_score']:.1f}/5.0")
            print(f"🏥 Здоровье: {'Здоровое' if quality_result['is_healthy'] else 'Болезнь'}")
            
            # Урожайность
            yield_result = result['yield_prediction']
            print(f"📊 Урожайность: {yield_result['predicted_yield_tons_per_ha']:.2f} т/га")
            
            # Комплексные рекомендации
            if result['comprehensive_recommendations']:
                print("\n🎯 Комплексные рекомендации:")
                for i, rec in enumerate(result['comprehensive_recommendations'], 1):
                    print(f"   {i}. {rec}")
        else:
            print(f"❌ Ошибка комплексного анализа: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"❌ Ошибка при тестировании комплексного анализа: {e}")

def main():
    """Главная функция демонстрации"""
    print("🌾 Демонстрация Агропайплайн API")
    print("=" * 50)
    
    # Проверяем состояние API
    if not test_api_health():
        print("\n❌ API недоступен. Запустите сервер командой: python run_api.py")
        return
    
    # Тестируем все функции
    test_classification()
    test_quality_assessment() 
    test_yield_prediction()
    test_comprehensive_analysis()
    
    print("\n✅ Демонстрация завершена!")
    print("🌐 Попробуйте веб-интерфейс: python run_frontend.py")

if __name__ == "__main__":
    main() 