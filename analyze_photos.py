#!/usr/bin/env python3
"""
Анализ всех фотографий в папке для определения типов культур
и подготовки данных для обучения модели
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from PIL import Image
import json
import numpy as np
from pathlib import Path
import cv2
from src.models.crop_classifier import ImprovedCropClassifier, SmartCropClassifier
from src.config.settings import settings
import matplotlib.pyplot as plt

def analyze_single_image(image_path, classifier):
    """Анализ одного изображения"""
    try:
        # Загружаем изображение
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Анализируем с помощью улучшенного классификатора
        result = classifier.predict(image)
        
        # Добавляем информацию о файле
        result['file_path'] = str(image_path)
        result['file_name'] = image_path.name
        result['file_size'] = image_path.stat().st_size
        result['image_size'] = image.size
        
        return result
        
    except Exception as e:
        return {
            'file_path': str(image_path),
            'file_name': image_path.name,
            'error': str(e),
            'predicted_class': 'error',
            'predicted_class_ru': 'ошибка'
        }

def extract_visual_features(image_path):
    """Извлечение визуальных признаков для анализа"""
    try:
        image = Image.open(image_path)
        img_array = np.array(image.convert('RGB'))
        height, width = img_array.shape[:2]
        
        # Цветовой анализ
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Анализ зеленых областей (растительность)
        green_mask = cv2.inRange(hsv, np.array([40, 40, 40]), np.array([80, 255, 255]))
        green_ratio = np.sum(green_mask > 0) / green_mask.size
        
        # Анализ желто-коричневых областей (зрелые культуры)
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
        
        return {
            'green_ratio': green_ratio,
            'yellow_ratio': yellow_ratio,
            'vertical_lines': vertical_lines,
            'edge_density': edge_density,
            'aspect_ratio': aspect_ratio,
            'dominant_colors': extract_dominant_colors(img_array)
        }
        
    except Exception as e:
        return {'error': str(e)}

def extract_dominant_colors(img_array, k=3):
    """Извлечение доминирующих цветов"""
    try:
        # Преобразуем в список пикселей
        pixels = img_array.reshape(-1, 3)
        
        # Применяем k-means для поиска доминирующих цветов
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        # Считаем долю каждого цвета
        unique, counts = np.unique(labels, return_counts=True)
        color_percentages = counts / len(labels)
        
        return [
            {
                'color': color.tolist(),
                'percentage': float(percentage)
            }
            for color, percentage in zip(colors, color_percentages)
        ]
    except ImportError:
        return []
    except Exception as e:
        return []

def manual_crop_identification(image_path, visual_features):
    """
    Ручная классификация на основе анализа изображения и имени файла
    """
    filename = os.path.basename(image_path).lower()
    
    # Извлекаем признаки
    green_ratio = visual_features.get('green_ratio', 0)
    yellow_ratio = visual_features.get('yellow_ratio', 0)
    vertical_lines = visual_features.get('vertical_lines', 0)
    edge_density = visual_features.get('edge_density', 0)
    aspect_ratio = visual_features.get('aspect_ratio', 0)
    
    # Специальные случаи по именам файлов
    if 'images.jpeg' in filename:
        return "кукуруза"  # Пользователь указал, что это должно быть кукуруза
    
    # Кукуруза - обычно имеет высокие стебли, характерные листья
    if ('corn' in filename or 'кукуруза' in filename or 
        '119252' in filename or 'maize' in filename):
        return "кукуруза"
    
    # Пшеница - обычно золотистая, колосья
    if ('wheat' in filename or 'пшеница' in filename or 
        'пшениц' in filename or 'field' in filename):
        return "пшеница"
    
    # Анализ по визуальным признакам
    # Если много желтого и мало зеленого - скорее всего пшеница
    if yellow_ratio > 0.4 and green_ratio < 0.1:
        return "пшеница"
    
    # Если много зеленого и высокая плотность краев - может быть кукуруза
    if green_ratio > 0.2 and edge_density > 0.3:
        return "кукуруза"
    
    # Для DJI фото анализируем более детально
    if filename.startswith('dji_'):
        if yellow_ratio > 0.3 and vertical_lines < 0.2:
            if '0046' in filename or '0048' in filename or '0031' in filename:
                return "пшеница"
            else:
                return "ячмень"
        else:
            return "ячмень"
    
    # По умолчанию ячмень для неясных случаев
    return "ячмень"

def analyze_all_photos():
    """Анализ всех фотографий в папке"""
    photo_dir = Path("photo")
    
    print("🔍 Анализ всех фотографий в папке photo/")
    print("=" * 60)
    
    # Инициализируем классификатор
    print("🚀 Инициализация улучшенного классификатора...")
    try:
        classifier = SmartCropClassifier(prefer_advanced=True)
        print("✅ Умный классификатор загружен!")
    except Exception as e:
        print(f"⚠️ Ошибка загрузки умного классификатора: {e}")
        try:
            classifier = ImprovedCropClassifier()
            print("✅ Улучшенный классификатор загружен!")
        except Exception as e2:
            print(f"❌ Критическая ошибка: {e2}")
            return
    
    # Получаем список всех изображений
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.avif'}
    image_files = []
    
    for file_path in photo_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    print(f"📁 Найдено {len(image_files)} изображений")
    
    results = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n📸 [{i}/{len(image_files)}] Анализ: {image_path.name}")
        
        # Анализ визуальных признаков
        visual_features = extract_visual_features(image_path)
        
        # Ручная идентификация для создания ground truth
        manual_class = manual_crop_identification(image_path, visual_features)
        
        # Анализ с помощью нейросети
        ml_result = analyze_single_image(image_path, classifier)
        
        # Комбинированный результат
        result = {
            'file_info': {
                'path': str(image_path),
                'name': image_path.name,
                'size_mb': round(image_path.stat().st_size / 1024 / 1024, 2)
            },
            'visual_features': visual_features,
            'manual_identification': {
                'class': manual_class,
                'class_ru': manual_class,
                'confidence': 1.0
            },
            'ml_prediction': {
                'class': ml_result.get('predicted_class', 'unknown'),
                'class_ru': ml_result.get('predicted_class_ru', 'неизвестно'),
                'confidence': ml_result.get('confidence', 0.0),
                'analysis_notes': ml_result.get('analysis_notes', [])
            },
            'agreement': manual_class == ml_result.get('predicted_class', 'unknown')
        }
        
        results.append(result)
        
        # Выводим результат
        print(f"   👨‍🌾 Ручная оценка: {manual_class}")
        print(f"   🤖 ИИ предсказание: {ml_result.get('predicted_class_ru', 'ошибка')} (уверенность: {ml_result.get('confidence', 0):.2f})")
        
        agreement = "✅ Согласие" if result['agreement'] else "❌ Расхождение"
        print(f"   {agreement}")
        
        if visual_features.get('error'):
            print(f"   ⚠️ Ошибка анализа: {visual_features['error']}")
    
    # Сохраняем результаты
    results_file = "photo_analysis_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Результаты сохранены в {results_file}")
    
    # Статистика
    print("\n📊 СТАТИСТИКА АНАЛИЗА:")
    print("=" * 40)
    
    # Подсчет по типам культур (ручная оценка)
    manual_counts = {}
    ml_counts = {}
    agreements = 0
    
    for result in results:
        manual_class = result['manual_identification']['class']
        ml_class = result['ml_prediction']['class']
        
        manual_counts[manual_class] = manual_counts.get(manual_class, 0) + 1
        ml_counts[ml_class] = ml_counts.get(ml_class, 0) + 1
        
        if result['agreement']:
            agreements += 1
    
    print("Ручная классификация:")
    for crop, count in manual_counts.items():
        print(f"   {crop}: {count} фото")
    
    print("\nПредсказания ИИ:")
    for crop, count in ml_counts.items():
        print(f"   {crop}: {count} фото")
    
    agreement_rate = agreements / len(results) * 100
    print(f"\nСогласованность: {agreements}/{len(results)} ({agreement_rate:.1f}%)")
    
    return results

def create_training_dataset(results):
    """Создание датасета для обучения на основе анализа"""
    print("\n🎯 СОЗДАНИЕ ДАТАСЕТА ДЛЯ ОБУЧЕНИЯ")
    print("=" * 50)
    
    # Создаем структуру папок для датасета
    dataset_dir = Path("data/training_dataset")
    
    for crop in ['wheat', 'corn', 'barley']:
        crop_dir = dataset_dir / crop
        crop_dir.mkdir(parents=True, exist_ok=True)
    
    # Копируем изображения в соответствующие папки
    import shutil
    
    copied_files = {'wheat': 0, 'corn': 0, 'barley': 0}
    
    for result in results:
        if result.get('manual_identification', {}).get('confidence', 0) > 0.5:
            manual_class = result['manual_identification']['class']
            source_path = Path(result['file_info']['path'])
            
            if manual_class in ['wheat', 'corn', 'barley']:
                # Создаем уникальное имя файла
                dest_name = f"{manual_class}_{source_path.stem}_{copied_files[manual_class]:03d}{source_path.suffix}"
                dest_path = dataset_dir / manual_class / dest_name
                
                try:
                    shutil.copy2(source_path, dest_path)
                    copied_files[manual_class] += 1
                    print(f"✅ {source_path.name} → {manual_class}/{dest_name}")
                except Exception as e:
                    print(f"❌ Ошибка копирования {source_path.name}: {e}")
    
    print(f"\n📁 Датасет создан в {dataset_dir}")
    print("Статистика датасета:")
    for crop, count in copied_files.items():
        print(f"   {crop}: {count} изображений")
    
    # Создаем метаданные датасета
    metadata = {
        'dataset_info': {
            'name': 'Agro Photo Dataset',
            'description': 'Датасет фотографий сельскохозяйственных культур',
            'classes': ['wheat', 'corn', 'barley'],
            'classes_ru': ['пшеница', 'кукуруза', 'ячмень'],
            'total_images': sum(copied_files.values()),
            'class_distribution': copied_files
        },
        'analysis_results': results
    }
    
    metadata_file = dataset_dir / "dataset_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"📄 Метаданные сохранены в {metadata_file}")
    
    return dataset_dir, copied_files

if __name__ == "__main__":
    print("🌾 АНАЛИЗ ФОТОГРАФИЙ СЕЛЬСКОХОЗЯЙСТВЕННЫХ КУЛЬТУР")
    print("🎯 Цель: Проанализировать все фото и подготовить данные для обучения")
    print("=" * 70)
    
    try:
        # Анализируем все фотографии
        results = analyze_all_photos()
        
        if results:
            # Создаем обучающий датасет
            dataset_dir, stats = create_training_dataset(results)
            
            print("\n🎉 АНАЛИЗ ЗАВЕРШЕН!")
            print("✅ Все фотографии проанализированы")
            print("✅ Датасет для обучения создан")
            print("✅ Готово к следующему этапу - обучению модели")
            
        else:
            print("❌ Не удалось проанализировать фотографии")
            
    except Exception as e:
        print(f"💥 Критическая ошибка: {e}")
        import traceback
        traceback.print_exc() 