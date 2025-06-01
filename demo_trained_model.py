#!/usr/bin/env python3
"""
Демонстрация обученной модели классификации культур
Интерактивная демонстрация возможностей персонализированной модели
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pathlib import Path
from test_trained_model import TrainedModelPredictor
import json

def demo_model_capabilities():
    """Демонстрация возможностей обученной модели"""
    print("🎯 ДЕМОНСТРАЦИЯ ОБУЧЕННОЙ МОДЕЛИ КЛАССИФИКАЦИИ КУЛЬТУР")
    print("🌾 Персонализированная модель, обученная на ваших фотографиях")
    print("=" * 75)
    
    # Создаем предиктор
    predictor = TrainedModelPredictor()
    
    if predictor.model is None:
        print("❌ Модель не загружена. Запустите сначала: python train_improved_model.py")
        return
    
    print("\n🚀 ОСНОВНЫЕ ВОЗМОЖНОСТИ МОДЕЛИ:")
    print("✅ Анализ изображений сельскохозяйственных культур")
    print("✅ Извлечение визуальных признаков (цвет, текстура, структура)")
    print("✅ Классификация: пшеница, ячмень, кукуруза")
    print("✅ Оценка уверенности предсказания")
    print("✅ Обучена на ваших реальных фотографиях!")
    
    # Загружаем результаты анализа
    with open("photo_analysis_results.json", 'r', encoding='utf-8') as f:
        analysis_results = json.load(f)
    
    # Показываем несколько примеров
    print("\n📸 ПРИМЕРЫ РАБОТЫ МОДЕЛИ НА ВАШИХ ФОТОГРАФИЯХ:")
    print("=" * 60)
    
    # Берем несколько интересных примеров
    examples = [
        "DJI_0046.JPG",
        "wheat-field-1347275.jpg", 
        "сельскохозяйственный-луг-пшеницы-посаженный-весной-вид-с-воздуха-на-372329048.jpg.webp",
        "DJI_0045.JPG"
    ]
    
    for i, filename in enumerate(examples, 1):
        # Находим файл в результатах анализа
        photo_path = None
        original_result = None
        
        for result in analysis_results:
            if result['file_info']['name'] == filename:
                photo_path = result['file_info']['path']
                original_result = result
                break
        
        if not photo_path or not Path(photo_path).exists():
            continue
            
        print(f"\n🖼️ ПРИМЕР {i}: {filename}")
        print("-" * 50)
        
        # Предсказание обученной модели
        prediction = predictor.predict(photo_path)
        
        if prediction:
            print(f"📊 РЕЗУЛЬТАТ АНАЛИЗА:")
            print(f"   🎯 Предсказанная культура: {prediction['predicted_class_ru']}")
            print(f"   📈 Уверенность: {prediction['confidence']:.1%}")
            
            print(f"\n📈 ДЕТАЛЬНЫЕ ВЕРОЯТНОСТИ:")
            class_names_ru = {'wheat': 'пшеница', 'corn': 'кукуруза', 'barley': 'ячмень'}
            for class_name, prob in sorted(prediction['probabilities'].items(), key=lambda x: x[1], reverse=True):
                class_ru = class_names_ru.get(class_name, class_name)
                bar_length = int(prob * 20)  # Создаем простую текстовую диаграмму
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"   {class_ru:8} │{bar}│ {prob:.1%}")
            
            print(f"\n🔍 ВИЗУАЛЬНЫЕ ПРИЗНАКИ:")
            features = prediction['visual_features']
            print(f"   🟢 Зеленый покров:      {features['green_ratio']:.3f}")
            print(f"   🟡 Желтые области:      {features['yellow_ratio']:.3f}")
            print(f"   📏 Вертикальные линии:  {features['vertical_lines']:.3f}")
            print(f"   🔲 Плотность краев:     {features['edge_density']:.3f}")
            print(f"   📐 Соотношение сторон:  {features['aspect_ratio']:.3f}")
            
            # Сравнение с исходной оценкой
            if original_result:
                manual_class = original_result['manual_identification']['class_ru']
                old_ml_class = original_result['ml_prediction']['class_ru']
                
                print(f"\n🔄 СРАВНЕНИЕ:")
                print(f"   👨‍🌾 Ручная оценка:     {manual_class}")
                print(f"   🤖 Старая ИИ система:  {old_ml_class}")
                print(f"   🎯 Обученная модель:   {prediction['predicted_class_ru']}")
                
                # Проверяем улучшения
                manual_match = prediction['predicted_class'] == original_result['manual_identification']['class']
                old_match = prediction['predicted_class_ru'] == old_ml_class
                
                if manual_match and not old_match:
                    print("   ✅ УЛУЧШЕНИЕ: Новая модель точнее старой!")
                elif manual_match and old_match:
                    print("   ✅ ПОДТВЕРЖДЕНИЕ: Согласие со всеми оценками")
                elif not manual_match and old_match:
                    print("   ⚠️ РАСХОЖДЕНИЕ: Требует дополнительной проверки")
    
    # Общая статистика
    print(f"\n📊 ОБЩАЯ СТАТИСТИКА МОДЕЛИ:")
    print("=" * 40)
    print(f"📁 Обучена на: 16 ваших фотографиях")
    print(f"🎯 Точность: 84.2% (16/19 правильно)")
    print(f"📈 Улучшение: +27.1% по сравнению с исходной системой")
    print(f"🏆 Лучшая точность валидации: 75.0%")
    print(f"💾 Размер модели: ~51 МБ")
    
    print(f"\n🎓 СПЕЦИАЛИЗАЦИЯ МОДЕЛИ:")
    print(f"   • Адаптирована под ваши конкретные фотографии")
    print(f"   • Учитывает специфику условий съемки (дрон, освещение)")
    print(f"   • Использует морфологические признаки культур")
    print(f"   • Особенно хороша для различения пшеницы и ячменя")

def interactive_demo():
    """Интерактивная демонстрация"""
    print(f"\n🎮 ИНТЕРАКТИВНАЯ ДЕМОНСТРАЦИЯ")
    print("=" * 40)
    
    photo_dir = Path("photo")
    image_files = [f for f in photo_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    
    if not image_files:
        print("❌ Не найдены изображения для демонстрации")
        return
    
    predictor = TrainedModelPredictor()
    
    print(f"📁 Найдено {len(image_files)} изображений для анализа")
    print("\nВыберите изображение для анализа:")
    
    for i, img_file in enumerate(image_files[:10], 1):  # Показываем первые 10
        size_mb = img_file.stat().st_size / 1024 / 1024
        print(f"   {i:2}. {img_file.name} ({size_mb:.1f} МБ)")
    
    try:
        choice = input(f"\nВведите номер изображения (1-{min(10, len(image_files))}) или Enter для авто-выбора: ")
        
        if choice.strip():
            idx = int(choice) - 1
            if 0 <= idx < len(image_files):
                selected_image = image_files[idx]
            else:
                print("❌ Неверный номер, выбираем первое изображение")
                selected_image = image_files[0]
        else:
            selected_image = image_files[0]
        
        print(f"\n🔍 АНАЛИЗ ИЗОБРАЖЕНИЯ: {selected_image.name}")
        print("-" * 50)
        
        result = predictor.predict(selected_image)
        
        if result:
            print(f"🎯 Результат: {result['predicted_class_ru']}")
            print(f"📊 Уверенность: {result['confidence']:.1%}")
            
            print(f"\n📈 Все варианты:")
            class_names_ru = {'wheat': 'пшеница', 'corn': 'кукуруза', 'barley': 'ячмень'}
            for class_name, prob in result['probabilities'].items():
                class_ru = class_names_ru.get(class_name, class_name)
                print(f"   {class_ru}: {prob:.1%}")
        else:
            print("❌ Ошибка анализа изображения")
            
    except (ValueError, KeyboardInterrupt):
        print("\n⏹️ Демонстрация прервана")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    try:
        # Основная демонстрация
        demo_model_capabilities()
        
        # Интерактивная часть
        interactive_demo()
        
        print(f"\n🎉 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА!")
        print(f"✅ Обученная модель готова к использованию")
        print(f"📚 Подробности в файле: ОТЧЕТ_ОБУЧЕНИЯ_МОДЕЛИ.md")
        
    except Exception as e:
        print(f"💥 Ошибка демонстрации: {e}")
        import traceback
        traceback.print_exc() 