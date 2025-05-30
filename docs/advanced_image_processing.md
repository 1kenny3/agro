# Продвинутая обработка изображений для нейронных сетей

Модуль `src/preprocessing/advanced_image_processor.py` предоставляет комплексные инструменты для обработки изображений сельскохозяйственных культур с целью улучшения работы нейронных сетей.

## Основные возможности

### 1. Автоматическое улучшение качества изображений
- Коррекция неравномерного освещения
- Адаптивное улучшение контраста (CLAHE)
- Шумоподавление с сохранением деталей
- Улучшение цветов растительности
- Умное повышение резкости

### 2. Оценка качества изображений
- Анализ резкости (Laplacian variance)
- Оценка контраста и яркости
- Определение уровня шума и размытия
- Анализ гистограммы и динамического диапазона
- Автоматические рекомендации по улучшению

### 3. Специализированные аугментации
- Сезонные вариации (весна, лето, осень)
- Имитация различных погодных условий
- Геометрические трансформации
- Цветовые коррекции

### 4. Мультимасштабные признаки
- Обработка изображений на разных масштабах
- Карты внимания для выделения важных областей
- Текстурный анализ с Gabor фильтрами
- Индексы растительности (псевдо-NDVI)

## Классы и их использование

### AdvancedImageEnhancer

Основной класс для улучшения качества изображений.

```python
from src.preprocessing.advanced_image_processor import AdvancedImageEnhancer
from PIL import Image

enhancer = AdvancedImageEnhancer()
image = Image.open("path/to/image.jpg")
enhanced_image = enhancer.enhance_for_recognition(image)
```

**Методы:**
- `enhance_for_recognition(image)` - комплексное улучшение изображения
- `_correct_illumination(img_array)` - коррекция освещения
- `_adaptive_contrast_enhancement(img_array)` - улучшение контраста
- `_denoise_preserve_details(img_array)` - шумоподавление
- `_enhance_vegetation_colors(img_array)` - улучшение цветов растений
- `_smart_sharpening(img_array)` - умное повышение резкости

### QualityAwarePreprocessor

Препроцессор с оценкой качества изображения.

```python
from src.preprocessing.advanced_image_processor import QualityAwarePreprocessor

processor = QualityAwarePreprocessor(target_size=224)

# Оценка качества
quality_info = processor.assess_image_quality(image)
print(f"Качество: {quality_info['quality_level']}")
print(f"Оценка: {quality_info['quality_score']:.3f}")

# Адаптивная предобработка
result = processor.adaptive_preprocess(image)
tensor = result['tensor']  # Готовый тензор для модели
```

**Основные метрики качества:**
- `sharpness` - резкость (чем выше, тем лучше)
- `contrast` - контраст (оптимально 30-80)
- `brightness` - яркость (оптимально 80-180)
- `noise_level` - уровень шума (чем ниже, тем лучше)
- `blur_level` - уровень размытия (чем ниже, тем лучше)
- `quality_score` - общая оценка (0-1, чем выше, тем лучше)

### AgriculturalAugmentation

Специализированные аугментации для сельскохозяйственных изображений.

```python
from src.preprocessing.advanced_image_processor import AgriculturalAugmentation

augmenter = AgriculturalAugmentation()

# Сезонные вариации
seasonal_images = augmenter.create_seasonal_variations(image)
spring_img, summer_img, autumn_img = seasonal_images

# Аугментации для обучения
augmented_images = augmenter.augment_for_training(image, num_augmentations=5)
```

**Типы аугментаций:**
- Геометрические: повороты, отражения, сдвиги
- Цветовые: изменение яркости, контраста, насыщенности
- Атмосферные: туман, блики, тени
- Шум и размытие: различные типы искажений
- Сельскохозяйственные: обрезка, выпадающие области

### FeatureEnhancedPreprocessor

Комплексный препроцессор с расширенными возможностями.

```python
from src.preprocessing.advanced_image_processor import FeatureEnhancedPreprocessor

processor = FeatureEnhancedPreprocessor(target_size=224)

# Комплексная обработка
result = processor.comprehensive_preprocess(image, include_augmentations=True)

# Доступные данные:
tensor = result['tensor']  # Основной тензор
multi_scale = result['multi_scale_tensors']  # Мультимасштабные тензоры
attention_maps = result['attention_maps']  # Карты внимания
quality_info = result['quality_info']  # Информация о качестве
```

**Карты внимания:**
- `texture_attention` - карта текстур (Gabor фильтры)
- `vegetation_attention` - карта растительности (псевдо-NDVI)
- `edge_attention` - карта краев (Canny)
- `combined_attention` - объединенная карта

## Быстрое использование

Для быстрого доступа к основной функциональности используйте функцию `enhance_image_for_neural_network`:

```python
from src.preprocessing.advanced_image_processor import enhance_image_for_neural_network
from PIL import Image

image = Image.open("path/to/image.jpg")

# Базовая обработка
result = enhance_image_for_neural_network(image)
tensor = result['tensor']

# Полная обработка с дополнительными возможностями
result = enhance_image_for_neural_network(
    image,
    target_size=224,
    include_multi_scale=True,
    include_attention=True
)

# Доступные данные
main_tensor = result['tensor']
multi_scale_tensors = result['multi_scale_tensors']
attention_maps = result['attention_maps']
quality_score = result['quality_info']['quality_score']
```

## Интеграция с существующими моделями

### Использование с PyTorch моделями

```python
import torch
from torch.utils.data import Dataset, DataLoader

class EnhancedCropDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = FeatureEnhancedPreprocessor()
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        
        # Обработка изображения
        result = self.processor.comprehensive_preprocess(image)
        
        return {
            'image': result['tensor'],
            'multi_scale': result['multi_scale_tensors'],
            'attention': result['attention_maps']['combined_attention'],
            'label': self.labels[idx],
            'quality': result['quality_info']['quality_score']
        }
```

### Использование с моделями классификации

```python
from src.models.crop_classifier import CropClassifier

# Инициализация модели
classifier = CropClassifier()

# Обработка изображения
result = enhance_image_for_neural_network(image)

# Предсказание
prediction = classifier.predict_single(result['tensor'])
print(f"Культура: {prediction['crop_class']}")
print(f"Уверенность: {prediction['confidence']:.3f}")
```

## Настройка параметров

### Пороги качества

Вы можете настроить пороги для определения необходимости улучшения:

```python
processor = QualityAwarePreprocessor()

# Изменение порога для применения улучшений
def custom_adaptive_preprocess(self, image, quality_threshold=0.6):
    quality_info = self.assess_image_quality(image)
    
    if quality_info["quality_score"] < quality_threshold:
        enhanced_image = self.enhancer.enhance_for_recognition(image)
    else:
        enhanced_image = image
    
    # ... остальная обработка
```

### Настройка аугментаций

```python
# Создание кастомных аугментаций
import albumentations as A

custom_augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, p=0.5),
    # Добавьте свои трансформации
])

augmenter = AgriculturalAugmentation()
augmenter.base_augmentations = custom_augmentations
```

## Производительность и оптимизация

### Рекомендации по использованию

1. **Пакетная обработка**: Обрабатывайте изображения пакетами для лучшей производительности
2. **Кэширование**: Сохраняйте результаты обработки для повторного использования
3. **Параллелизация**: Используйте многопоточность для обработки больших датасетов

```python
from concurrent.futures import ThreadPoolExecutor
import pickle

def process_and_cache_images(image_paths, cache_dir="processed_cache"):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    
    processor = FeatureEnhancedPreprocessor()
    
    def process_single(image_path):
        cache_file = cache_dir / f"{Path(image_path).stem}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        image = Image.open(image_path)
        result = processor.comprehensive_preprocess(image)
        
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_single, image_paths))
    
    return results
```

## Примеры использования

### Пример 1: Базовое улучшение изображения

```python
from PIL import Image
from src.preprocessing.advanced_image_processor import AdvancedImageEnhancer

# Загрузка изображения
image = Image.open("crop_image.jpg")

# Улучшение
enhancer = AdvancedImageEnhancer()
enhanced = enhancer.enhance_for_recognition(image)

# Сохранение результата
enhanced.save("enhanced_crop_image.jpg")
```

### Пример 2: Оценка качества датасета

```python
from pathlib import Path
from src.preprocessing.advanced_image_processor import QualityAwarePreprocessor

processor = QualityAwarePreprocessor()
image_dir = Path("data/raw/images")

quality_scores = []
for img_path in image_dir.glob("*.jpg"):
    image = Image.open(img_path)
    quality = processor.assess_image_quality(image)
    quality_scores.append({
        'filename': img_path.name,
        'quality_score': quality['quality_score'],
        'quality_level': quality['quality_level'],
        'recommendations': quality['recommendations']
    })

# Сортировка по качеству
quality_scores.sort(key=lambda x: x['quality_score'], reverse=True)

print("Топ-5 изображений по качеству:")
for item in quality_scores[:5]:
    print(f"{item['filename']}: {item['quality_score']:.3f} ({item['quality_level']})")
```

### Пример 3: Создание аугментированного датасета

```python
from src.preprocessing.advanced_image_processor import AgriculturalAugmentation

augmenter = AgriculturalAugmentation()
input_dir = Path("data/raw/wheat")
output_dir = Path("data/augmented/wheat")
output_dir.mkdir(parents=True, exist_ok=True)

for img_path in input_dir.glob("*.jpg"):
    image = Image.open(img_path)
    
    # Создаем аугментации
    augmented_images = augmenter.augment_for_training(image, num_augmentations=5)
    seasonal_images = augmenter.create_seasonal_variations(image)
    
    # Сохраняем оригинал
    stem = img_path.stem
    image.save(output_dir / f"{stem}_original.jpg")
    
    # Сохраняем аугментации
    for i, aug_img in enumerate(augmented_images):
        aug_img.save(output_dir / f"{stem}_aug_{i}.jpg")
    
    # Сохраняем сезонные вариации
    seasons = ["spring", "summer", "autumn"]
    for season_img, season in zip(seasonal_images, seasons):
        season_img.save(output_dir / f"{stem}_{season}.jpg")
```

## Заключение

Модуль продвинутой обработки изображений предоставляет комплексные инструменты для улучшения качества работы нейронных сетей с сельскохозяйственными изображениями. Он включает в себя:

- Автоматическую оценку и улучшение качества изображений
- Специализированные аугментации для сельского хозяйства
- Мультимасштабную обработку и карты внимания
- Простые в использовании API для интеграции с существующими проектами

Использование этих инструментов поможет значительно улучшить точность распознавания культур, оценки качества и детекции болезней в ваших моделях машинного обучения. 