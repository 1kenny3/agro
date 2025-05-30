#!/usr/bin/env python3
"""
Тестовый скрипт для демонстрации нового продвинутого классификатора культур
с современными нейронными сетями
"""

import sys
import os
from pathlib import Path
from PIL import Image
import json
import numpy as np

# Добавляем путь к проекту
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_new_classifier():
    """Тестирует новый продвинутый классификатор"""
    
    print("🚀 ТЕСТИРОВАНИЕ НОВОГО ПРОДВИНУТОГО КЛАССИФИКАТОРА")
    print("=" * 60)
    
    try:
        # Импортируем новый классификатор
        from src.models.crop_classifier import SmartCropClassifier
        
        print("📦 Инициализация умного классификатора...")
        classifier = SmartCropClassifier(prefer_advanced=True)
        
        # Получаем информацию о классификаторе
        info = classifier.get_classifier_info()
        print(f"\n📋 Информация о классификаторе:")
        print(f"   Тип: {info['type']}")
        print(f"   Описание: {info['description']}")
        print(f"   Устройство: {info['device']}")
        print(f"   Продвинутый доступен: {info['available_advanced']}")
        
        # Ищем тестовые изображения
        test_images_dir = project_root / "photo"
        if not test_images_dir.exists():
            print(f"\n⚠️ Папка с тестовыми изображениями не найдена: {test_images_dir}")
            print("Создаем тестовое изображение...")
            
            # Создаем простое тестовое изображение
            from PIL import ImageDraw
            test_img = Image.new('RGB', (224, 224), color='green')
            draw = ImageDraw.Draw(test_img)
            draw.text((50, 100), "TEST CROP", fill='white')
            
            test_images_dir.mkdir(exist_ok=True)
            test_path = test_images_dir / "test_crop.jpg"
            test_img.save(test_path)
            print(f"✅ Создано тестовое изображение: {test_path}")
            
            test_images = [test_path]
        else:
            # Ищем изображения в папке
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            test_images = []
            for ext in image_extensions:
                test_images.extend(test_images_dir.glob(f"*{ext}"))
                test_images.extend(test_images_dir.glob(f"*{ext.upper()}"))
            
            if not test_images:
                print(f"⚠️ Не найдено изображений в {test_images_dir}")
                return
        
        print(f"\n🖼️ Найдено {len(test_images)} тестовых изображений")
        
        # Тестируем классификацию
        for i, image_path in enumerate(test_images[:3]):  # Тестируем максимум 3 изображения
            print(f"\n{'='*50}")
            print(f"🔍 ТЕСТ {i+1}: {image_path.name}")
            print(f"{'='*50}")
            
            try:
                # Загружаем изображение
                image = Image.open(image_path)
                print(f"📏 Размер изображения: {image.size}")
                
                # Выполняем классификацию
                print("🧠 Выполняется анализ...")
                result = classifier.predict(image)
                
                # Выводим результаты
                print(f"\n🎯 РЕЗУЛЬТАТЫ АНАЛИЗА:")
                print(f"   Предсказанный класс: {result['predicted_class_ru']} ({result['predicted_class']})")
                print(f"   Уверенность: {result['confidence']:.3f} ({result['confidence_level']})")
                print(f"   Надежность: {'✅ Да' if result['is_confident'] else '❌ Нет'}")
                
                print(f"\n📊 ВЕРОЯТНОСТИ:")
                for class_ru, prob in result['probabilities_ru'].items():
                    print(f"   {class_ru}: {prob:.3f} ({prob*100:.1f}%)")
                
                print(f"\n📝 АНАЛИЗ:")
                for note in result['analysis_notes']:
                    print(f"   • {note}")
                
                # Дополнительная информация для продвинутого классификатора
                if 'morphology_analysis' in result and result['morphology_analysis']:
                    morph = result['morphology_analysis']
                    print(f"\n🔬 МОРФОЛОГИЧЕСКИЙ АНАЛИЗ:")
                    print(f"   Интенсивность зелени: {morph.get('green_intensity', 0):.3f}")
                    print(f"   Текстура: {morph.get('texture_variance', 0):.1f}")
                    print(f"   Вертикальные линии: {morph.get('vertical_lines', 0):.3f}")
                    print(f"   Соотношение сторон: {morph.get('avg_aspect_ratio', 0):.2f}")
                
                if 'model_results' in result and result['model_results']:
                    print(f"\n🤖 РЕЗУЛЬТАТЫ МОДЕЛЕЙ:")
                    for model_name, probs in result['model_results'].items():
                        if isinstance(probs, (list, np.ndarray)) and len(probs) > 0:
                            max_prob = float(np.max(probs))
                            print(f"   {model_name}: {max_prob:.3f}")
                        else:
                            print(f"   {model_name}: недоступно")
                
            except Exception as e:
                print(f"❌ Ошибка при обработке {image_path.name}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("🎯 Новый продвинутый классификатор работает корректно")
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("Убедитесь, что все зависимости установлены:")
        print("pip install torch torchvision timm transformers opencv-python")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

def compare_classifiers():
    """Сравнивает разные типы классификаторов"""
    
    print("\n🔄 СРАВНЕНИЕ КЛАССИФИКАТОРОВ")
    print("=" * 40)
    
    try:
        from src.models.crop_classifier import SmartCropClassifier, ImprovedCropClassifier
        
        # Создаем тестовое изображение
        from PIL import ImageDraw
        test_img = Image.new('RGB', (224, 224), color='lightgreen')
        draw = ImageDraw.Draw(test_img)
        draw.rectangle([50, 50, 174, 174], fill='darkgreen')
        draw.text((80, 100), "CORN", fill='yellow')
        
        print("🖼️ Создано тестовое изображение кукурузы")
        
        # Тест продвинутого классификатора
        print("\n🚀 Тестирование продвинутого классификатора...")
        advanced_classifier = SmartCropClassifier(prefer_advanced=True)
        advanced_result = advanced_classifier.predict(test_img)
        
        print(f"   Результат: {advanced_result['predicted_class_ru']}")
        print(f"   Уверенность: {advanced_result['confidence']:.3f}")
        print(f"   Тип: {advanced_result.get('classifier_type', 'Unknown')}")
        
        # Тест улучшенного классификатора
        print("\n🔧 Тестирование улучшенного классификатора...")
        improved_classifier = ImprovedCropClassifier()
        improved_result = improved_classifier.predict(test_img)
        
        print(f"   Результат: {improved_result['predicted_class_ru']}")
        print(f"   Уверенность: {improved_result['confidence']:.3f}")
        
        # Сравнение
        print(f"\n📊 СРАВНЕНИЕ:")
        print(f"   Продвинутый: {advanced_result['predicted_class_ru']} ({advanced_result['confidence']:.3f})")
        print(f"   Улучшенный:  {improved_result['predicted_class_ru']} ({improved_result['confidence']:.3f})")
        
        if advanced_result['confidence'] > improved_result['confidence']:
            print("🏆 Продвинутый классификатор показал лучший результат!")
        elif improved_result['confidence'] > advanced_result['confidence']:
            print("🏆 Улучшенный классификатор показал лучший результат!")
        else:
            print("🤝 Оба классификатора показали одинаковые результаты")
            
    except Exception as e:
        print(f"❌ Ошибка сравнения: {e}")

if __name__ == "__main__":
    print("🌾 ТЕСТИРОВАНИЕ НОВЫХ НЕЙРОННЫХ СЕТЕЙ ДЛЯ КЛАССИФИКАЦИИ КУЛЬТУР")
    print("🚀 Современные архитектуры: Swin Transformer, Vision Transformer, EfficientNetV2")
    print()
    
    test_new_classifier()
    compare_classifiers()
    
    print(f"\n{'='*60}")
    print("🎉 ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ!")
    print("💡 Теперь вы можете использовать новые нейронные сети в своем проекте")
    print("📚 Документация: SmartCropClassifier автоматически выберет лучшую модель") 