#!/usr/bin/env python3
"""
Тест для проверки согласованности результатов между различными типами анализа
Проверяем что одно и то же изображение дает одинаковые результаты классификации
в разных эндпоинтах
"""

import requests
import io
from PIL import Image, ImageDraw
import time

API_BASE_URL = "http://localhost:8000"

def create_test_corn_image():
    """Создание тестового изображения кукурузы"""
    # Создаем изображение с характерными для кукурузы признаками
    image = Image.new('RGB', (512, 768), color='lightblue')  # Вертикальная ориентация
    draw = ImageDraw.Draw(image)
    
    # Рисуем зеленые вертикальные стебли
    for x in range(50, 450, 40):
        # Стебель
        draw.rectangle([x, 200, x+15, 700], fill='darkgreen')
        # Листья (широкие, характерные для кукурузы)
        for y in range(250, 650, 80):
            # Левый лист
            draw.polygon([
                (x-30, y), (x, y+10), (x, y+60), (x-25, y+50)
            ], fill='green')
            # Правый лист
            draw.polygon([
                (x+45, y), (x+15, y+10), (x+15, y+60), (x+40, y+50)
            ], fill='green')
    
    # Рисуем метелки в верхней части (характерный признак кукурузы)
    for x in range(60, 440, 40):
        for y in range(50, 150, 20):
            draw.ellipse([x-5, y, x+25, y+15], fill='yellow')
            draw.ellipse([x-3, y+2, x+23, y+13], fill='brown')
    
    return image

def test_endpoint(endpoint_url, image, test_name):
    """Тестирование конкретного эндпоинта"""
    print(f"\n🧪 Тестирование: {test_name}")
    print(f"📡 Эндпоинт: {endpoint_url}")
    
    # Подготавливаем изображение
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    try:
        files = {'file': ('test_corn.png', img_byte_arr, 'image/png')}
        response = requests.post(endpoint_url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            
            # Извлекаем результат классификации
            if 'crop_classification' in result:
                # Комплексный анализ
                crop_result = result['crop_classification']
                classification_result = {
                    'predicted_class': crop_result.get('predicted_class'),
                    'predicted_class_ru': crop_result.get('predicted_class_ru'),
                    'confidence': crop_result.get('confidence'),
                    'analysis_notes': crop_result.get('analysis_notes', [])
                }
            else:
                # Обычная классификация
                classification_result = {
                    'predicted_class': result.get('predicted_class'),
                    'predicted_class_ru': result.get('predicted_class_ru'),
                    'confidence': result.get('confidence'),
                    'analysis_notes': result.get('analysis_notes', [])
                }
            
            print(f"✅ Успешный ответ")
            print(f"🌾 Культура: {classification_result['predicted_class_ru']}")
            print(f"🎯 Уверенность: {classification_result['confidence']:.3f}")
            
            # Показываем ключевые заметки анализа
            if classification_result['analysis_notes']:
                key_notes = [note for note in classification_result['analysis_notes'][:3]]
                print("📝 Заметки анализа:")
                for note in key_notes:
                    print(f"   • {note}")
            
            return classification_result
            
        else:
            print(f"❌ Ошибка: {response.status_code}")
            print(f"📄 Ответ: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Исключение: {e}")
        return None

def compare_results(result1, result2, name1, name2):
    """Сравнение результатов двух эндпоинтов"""
    print(f"\n🔍 Сравнение результатов: {name1} vs {name2}")
    print("=" * 60)
    
    if not result1 or not result2:
        print("❌ Невозможно сравнить - один из результатов отсутствует")
        return False
    
    # Сравниваем основные поля
    class_match = result1['predicted_class'] == result2['predicted_class']
    class_ru_match = result1['predicted_class_ru'] == result2['predicted_class_ru']
    
    # Допускаем небольшое различие в уверенности (до 5%)
    confidence_diff = abs(result1['confidence'] - result2['confidence'])
    confidence_similar = confidence_diff <= 0.05
    
    print(f"🌾 Класс культуры: {'✅' if class_match else '❌'}")
    print(f"   {name1}: {result1['predicted_class']} ({result1['predicted_class_ru']})")
    print(f"   {name2}: {result2['predicted_class']} ({result2['predicted_class_ru']})")
    
    print(f"🎯 Уверенность: {'✅' if confidence_similar else '❌'}")
    print(f"   {name1}: {result1['confidence']:.3f}")
    print(f"   {name2}: {result2['confidence']:.3f}")
    print(f"   Разница: {confidence_diff:.3f}")
    
    overall_match = class_match and class_ru_match and confidence_similar
    
    if overall_match:
        print("🎉 РЕЗУЛЬТАТЫ СОГЛАСОВАНЫ!")
    else:
        print("⚠️ ОБНАРУЖЕНО РАСХОЖДЕНИЕ!")
    
    return overall_match

def main():
    """Основная функция тестирования"""
    print("🚀 Тест согласованности результатов классификации")
    print("=" * 60)
    print("Проверяем что одно изображение дает одинаковые результаты")
    print("в разных эндпоинтах API")
    
    # Создаем тестовое изображение кукурузы
    print("\n🌽 Создание тестового изображения кукурузы...")
    test_image = create_test_corn_image()
    
    # Определяем эндпоинты для тестирования
    endpoints = [
        (f"{API_BASE_URL}/classify", "Классификация культуры"),
        (f"{API_BASE_URL}/analyze", "Комплексный анализ"),
        (f"{API_BASE_URL}/analyze/comprehensive", "Расширенный комплексный анализ")
    ]
    
    results = {}
    
    # Тестируем каждый эндпоинт
    for endpoint_url, test_name in endpoints:
        result = test_endpoint(endpoint_url, test_image, test_name)
        results[test_name] = result
        time.sleep(1)  # Небольшая пауза между запросами
    
    # Сравниваем результаты попарно
    print("\n" + "=" * 60)
    print("📊 АНАЛИЗ СОГЛАСОВАННОСТИ")
    print("=" * 60)
    
    all_consistent = True
    
    # Сравниваем классификацию с комплексным анализом
    match1 = compare_results(
        results["Классификация культуры"], 
        results["Комплексный анализ"],
        "Классификация", "Комплексный анализ"
    )
    all_consistent = all_consistent and match1
    
    # Сравниваем классификацию с расширенным анализом
    match2 = compare_results(
        results["Классификация культуры"], 
        results["Расширенный комплексный анализ"],
        "Классификация", "Расширенный анализ"
    )
    all_consistent = all_consistent and match2
    
    # Сравниваем два типа комплексного анализа
    match3 = compare_results(
        results["Комплексный анализ"], 
        results["Расширенный комплексный анализ"],
        "Комплексный анализ", "Расширенный анализ"
    )
    all_consistent = all_consistent and match3
    
    # Финальный результат
    print("\n" + "=" * 60)
    print("🏁 ИТОГОВЫЙ РЕЗУЛЬТАТ")
    print("=" * 60)
    
    if all_consistent:
        print("🎉 ВСЕ ЭНДПОИНТЫ ДАЮТ СОГЛАСОВАННЫЕ РЕЗУЛЬТАТЫ!")
        print("✅ Проблема с различными типами анализа ИСПРАВЛЕНА")
    else:
        print("❌ ОБНАРУЖЕНЫ РАСХОЖДЕНИЯ В РЕЗУЛЬТАТАХ")
        print("⚠️ Требуется дополнительная диагностика")
    
    return all_consistent

if __name__ == "__main__":
    print("Убедитесь что API сервер запущен: python run_api.py")
    print("Нажмите Enter для продолжения...")
    input()
    
    success = main()
    
    if success:
        print("\n✅ Тест ПРОЙДЕН!")
    else:
        print("\n❌ Тест ПРОВАЛЕН!") 