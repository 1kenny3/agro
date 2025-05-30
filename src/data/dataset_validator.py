"""
Модуль для валидации качества датасета изображений
"""

import cv2
import numpy as np
from PIL import Image
import os
import json
from typing import Dict, List, Tuple
from pathlib import Path
import hashlib
from collections import defaultdict

from ..config.settings import settings

class DatasetValidator:
    """Класс для валидации качества датасета"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.min_resolution = 512
        self.max_file_size = 5 * 1024 * 1024  # 5MB
        self.min_file_size = 100 * 1024  # 100KB
        self.blur_threshold = 100  # Лапласиан для определения размытости
        
    def validate_image_quality(self, image_path: Path) -> Dict:
        """Проверка качества одного изображения"""
        results = {
            "path": str(image_path),
            "valid": True,
            "issues": [],
            "metrics": {}
        }
        
        try:
            # Проверка существования файла
            if not image_path.exists():
                results["valid"] = False
                results["issues"].append("Файл не найден")
                return results
            
            # Проверка размера файла
            file_size = image_path.stat().st_size
            results["metrics"]["file_size"] = file_size
            
            if file_size > self.max_file_size:
                results["issues"].append(f"Файл слишком большой: {file_size/1024/1024:.1f}MB")
            elif file_size < self.min_file_size:
                results["issues"].append(f"Файл слишком маленький: {file_size/1024:.1f}KB")
            
            # Загрузка изображения
            try:
                image = Image.open(image_path)
                image_cv = cv2.imread(str(image_path))
            except Exception as e:
                results["valid"] = False
                results["issues"].append(f"Ошибка открытия изображения: {e}")
                return results
            
            # Проверка разрешения
            width, height = image.size
            results["metrics"]["resolution"] = (width, height)
            
            if min(width, height) < self.min_resolution:
                results["issues"].append(f"Низкое разрешение: {width}x{height}")
            
            # Проверка формата
            if image.format not in ['JPEG', 'PNG']:
                results["issues"].append(f"Неподдерживаемый формат: {image.format}")
            
            # Проверка на размытость
            if image_cv is not None:
                gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
                results["metrics"]["blur_value"] = float(blur_value)
                
                if blur_value < self.blur_threshold:
                    results["issues"].append(f"Размытое изображение: {blur_value:.1f}")
            
            # Проверка экспозиции
            if image_cv is not None:
                mean_brightness = np.mean(gray)
                results["metrics"]["brightness"] = float(mean_brightness)
                
                if mean_brightness < 50:
                    results["issues"].append("Слишком темное изображение")
                elif mean_brightness > 200:
                    results["issues"].append("Слишком светлое изображение")
            
            # Проверка контрастности
            if image_cv is not None:
                contrast = np.std(gray)
                results["metrics"]["contrast"] = float(contrast)
                
                if contrast < 20:
                    results["issues"].append("Низкий контраст")
            
            if results["issues"]:
                results["valid"] = False
                
        except Exception as e:
            results["valid"] = False
            results["issues"].append(f"Неожиданная ошибка: {e}")
        
        return results
    
    def find_duplicates(self, image_paths: List[Path]) -> Dict[str, List[str]]:
        """Поиск дубликатов изображений по хешу"""
        hashes = {}
        duplicates = defaultdict(list)
        
        for path in image_paths:
            try:
                with open(path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
                if file_hash in hashes:
                    duplicates[file_hash].extend([str(hashes[file_hash]), str(path)])
                else:
                    hashes[file_hash] = path
                    
            except Exception as e:
                print(f"Ошибка при обработке {path}: {e}")
        
        return dict(duplicates)
    
    def validate_annotations(self, annotation_path: Path) -> Dict:
        """Проверка файла аннотаций"""
        results = {
            "path": str(annotation_path),
            "valid": True,
            "issues": []
        }
        
        if not annotation_path.exists():
            results["valid"] = False
            results["issues"].append("Файл аннотаций не найден")
            return results
        
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Проверка обязательных полей
            required_fields = ['filename', 'crop_type', 'health_status']
            for field in required_fields:
                if field not in data:
                    results["issues"].append(f"Отсутствует поле: {field}")
            
            # Проверка корректности значений
            if 'crop_type' in data:
                if data['crop_type'] not in settings.CROP_CLASSES:
                    results["issues"].append(f"Неизвестный тип культуры: {data['crop_type']}")
            
            if 'health_status' in data:
                valid_health = ['healthy', 'diseased', 'mixed']
                if data['health_status'] not in valid_health:
                    results["issues"].append(f"Неизвестный статус здоровья: {data['health_status']}")
            
            if results["issues"]:
                results["valid"] = False
                
        except json.JSONDecodeError as e:
            results["valid"] = False
            results["issues"].append(f"Ошибка парсинга JSON: {e}")
        
        return results
    
    def validate_dataset_structure(self) -> Dict:
        """Проверка структуры датасета"""
        results = {
            "valid": True,
            "issues": [],
            "statistics": {}
        }
        
        # Проверка основных директорий
        required_dirs = ['raw', 'processed', 'annotations']
        for dir_name in required_dirs:
            dir_path = self.dataset_path / dir_name
            if not dir_path.exists():
                results["issues"].append(f"Отсутствует директория: {dir_name}")
        
        # Статистика по культурам
        crop_stats = {}
        for crop in settings.CROP_CLASSES:
            crop_dir = self.dataset_path / 'raw' / crop
            if crop_dir.exists():
                images = list(crop_dir.glob('**/*.jpg')) + list(crop_dir.glob('**/*.png'))
                crop_stats[crop] = len(images)
            else:
                crop_stats[crop] = 0
                results["issues"].append(f"Отсутствует директория для {crop}")
        
        results["statistics"]["crop_counts"] = crop_stats
        results["statistics"]["total_images"] = sum(crop_stats.values())
        
        # Проверка минимального количества изображений
        min_images_per_crop = 100  # Минимальное количество
        for crop, count in crop_stats.items():
            if count < min_images_per_crop:
                results["issues"].append(f"Недостаточно изображений для {crop}: {count} < {min_images_per_crop}")
        
        if results["issues"]:
            results["valid"] = False
        
        return results
    
    def generate_quality_report(self) -> Dict:
        """Генерация полного отчета о качестве датасета"""
        print("🔍 Начинаем валидацию датасета...")
        
        report = {
            "dataset_path": str(self.dataset_path),
            "validation_timestamp": np.datetime64('now').isoformat(),
            "structure_validation": {},
            "image_validation": {},
            "annotation_validation": {},
            "duplicates": {},
            "summary": {}
        }
        
        # 1. Проверка структуры
        print("📁 Проверка структуры датасета...")
        report["structure_validation"] = self.validate_dataset_structure()
        
        # 2. Поиск всех изображений
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        all_images = []
        for ext in image_extensions:
            all_images.extend(self.dataset_path.glob(f'**/{ext}'))
            all_images.extend(self.dataset_path.glob(f'**/{ext.upper()}'))
        
        print(f"📸 Найдено {len(all_images)} изображений")
        
        # 3. Валидация изображений
        print("🔍 Проверка качества изображений...")
        valid_images = 0
        invalid_images = 0
        all_issues = []
        
        image_results = []
        for i, img_path in enumerate(all_images):
            if i % 100 == 0:
                print(f"Обработано {i}/{len(all_images)} изображений...")
            
            result = self.validate_image_quality(img_path)
            image_results.append(result)
            
            if result["valid"]:
                valid_images += 1
            else:
                invalid_images += 1
                all_issues.extend(result["issues"])
        
        report["image_validation"] = {
            "total_images": len(all_images),
            "valid_images": valid_images,
            "invalid_images": invalid_images,
            "results": image_results
        }
        
        # 4. Поиск дубликатов
        print("🔍 Поиск дубликатов...")
        duplicates = self.find_duplicates(all_images)
        report["duplicates"] = duplicates
        
        # 5. Проверка аннотаций
        print("📝 Проверка аннотаций...")
        annotation_files = list(self.dataset_path.glob('**/*.json'))
        annotation_results = []
        valid_annotations = 0
        
        for ann_path in annotation_files:
            result = self.validate_annotations(ann_path)
            annotation_results.append(result)
            if result["valid"]:
                valid_annotations += 1
        
        report["annotation_validation"] = {
            "total_annotations": len(annotation_files),
            "valid_annotations": valid_annotations,
            "results": annotation_results
        }
        
        # 6. Сводка
        report["summary"] = {
            "overall_valid": (
                report["structure_validation"]["valid"] and
                invalid_images == 0 and
                len(duplicates) == 0 and
                len(annotation_files) == valid_annotations
            ),
            "image_quality_score": valid_images / len(all_images) if all_images else 0,
            "annotation_completeness": valid_annotations / len(annotation_files) if annotation_files else 0,
            "duplicate_ratio": len(duplicates) / len(all_images) if all_images else 0,
            "recommendations": self._generate_recommendations(report)
        }
        
        print("✅ Валидация завершена!")
        return report
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Генерация рекомендаций по улучшению датасета"""
        recommendations = []
        
        # Анализ структуры
        struct_issues = report["structure_validation"]["issues"]
        if struct_issues:
            recommendations.append("Восстановите правильную структуру директорий датасета")
        
        # Анализ качества изображений
        invalid_ratio = report["image_validation"]["invalid_images"] / report["image_validation"]["total_images"]
        if invalid_ratio > 0.1:
            recommendations.append(f"Высокий процент некачественных изображений ({invalid_ratio:.1%}). Необходима дополнительная фильтрация")
        
        # Анализ дубликатов
        if len(report["duplicates"]) > 0:
            recommendations.append(f"Найдено {len(report['duplicates'])} групп дубликатов. Удалите повторяющиеся изображения")
        
        # Анализ количества данных по культурам
        crop_counts = report["structure_validation"]["statistics"]["crop_counts"]
        min_count = min(crop_counts.values()) if crop_counts else 0
        if min_count < 500:
            recommendations.append("Недостаточно данных для некоторых культур. Требуется дополнительный сбор изображений")
        
        # Баланс классов
        if crop_counts:
            max_count = max(crop_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            if imbalance_ratio > 3:
                recommendations.append("Сильный дисбаланс классов. Добавьте изображения для недопредставленных культур")
        
        return recommendations

def validate_dataset_quality(dataset_path: str) -> Dict:
    """Основная функция для валидации датасета"""
    validator = DatasetValidator(dataset_path)
    return validator.generate_quality_report()

if __name__ == "__main__":
    # Пример использования
    dataset_path = settings.DATA_DIR
    report = validate_dataset_quality(str(dataset_path))
    
    # Сохранение отчета
    report_path = dataset_path / "quality_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"📊 Отчет сохранен в {report_path}") 