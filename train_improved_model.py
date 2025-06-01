#!/usr/bin/env python3
"""
Обучение улучшенной модели классификации культур
на основе анализа фотографий из папки photo/
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import numpy as np
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.crop_classifier import CropClassifier
from src.config.settings import settings

class CropPhotoDataset(Dataset):
    """Датасет фотографий культур с визуальными признаками"""
    
    def __init__(self, image_paths, labels, visual_features_list, transform=None, use_features=True):
        self.image_paths = image_paths
        self.labels = labels
        self.visual_features_list = visual_features_list
        self.transform = transform
        self.use_features = use_features
        
        # Маппинг классов
        self.class_to_idx = {'wheat': 0, 'corn': 1, 'barley': 2}
        self.idx_to_class = {0: 'wheat', 1: 'corn', 2: 'barley'}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Загружаем изображение
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # Создаем черное изображение если не удалось загрузить
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        # Получаем метку
        label = self.class_to_idx.get(self.labels[idx], 2)  # По умолчанию ячмень
        
        # Визуальные признаки
        if self.use_features and idx < len(self.visual_features_list):
            features = self.visual_features_list[idx]
            visual_tensor = torch.tensor([
                features.get('green_ratio', 0.0),
                features.get('yellow_ratio', 0.0),
                features.get('vertical_lines', 0.0),
                features.get('edge_density', 0.0),
                features.get('aspect_ratio', 1.0)
            ], dtype=torch.float32)
        else:
            visual_tensor = torch.zeros(5, dtype=torch.float32)
        
        return image, visual_tensor, label

class EnhancedCropClassifier(nn.Module):
    """Улучшенная модель с учетом визуальных признаков"""
    
    def __init__(self, num_classes=3, model_name="efficientnet_b0", use_visual_features=True):
        super(EnhancedCropClassifier, self).__init__()
        
        self.use_visual_features = use_visual_features
        
        # Используем более простую модель для извлечения признаков
        import timm
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)  # num_classes=0 для feature extraction
        
        # Получаем размер выходных признаков
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_size = features.shape[1]
        
        print(f"🔧 Размер признаков backbone: {self.feature_size}")
        
        # Сеть для визуальных признаков
        if use_visual_features:
            self.visual_net = nn.Sequential(
                nn.Linear(5, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32)
            )
            
            # Комбинированный классификатор
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_size + 32, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, images, visual_features=None):
        # Извлекаем признаки изображения
        image_features = self.backbone(images)
        
        if self.use_visual_features and visual_features is not None:
            # Обрабатываем визуальные признаки
            visual_features = self.visual_net(visual_features)
            
            # Объединяем признаки
            combined_features = torch.cat([image_features, visual_features], dim=1)
        else:
            combined_features = image_features
        
        # Классификация
        output = self.classifier(combined_features)
        return output

def load_analyzed_data():
    """Загрузка проанализированных данных"""
    results_file = "photo_analysis_results.json"
    
    if not Path(results_file).exists():
        print(f"❌ Файл результатов {results_file} не найден!")
        print("Сначала запустите: python analyze_photos.py")
        return None, None, None
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    image_paths = []
    labels = []
    visual_features_list = []
    
    for result in results:
        # Пропускаем файлы с ошибками
        if result.get('visual_features', {}).get('error'):
            continue
            
        file_path = result['file_info']['path']
        
        # Определяем правильную метку на основе анализа
        manual_class = result['manual_identification']['class']
        manual_confidence = result['manual_identification']['confidence']
        
        # Используем только результаты с достаточной уверенностью
        if manual_confidence > 0.3:
            image_paths.append(file_path)
            labels.append(manual_class)
            visual_features_list.append(result['visual_features'])
    
    print(f"📊 Загружено {len(image_paths)} изображений для обучения")
    
    # Статистика по классам
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print("Распределение по классам:")
    for cls, count in class_counts.items():
        class_ru = {'wheat': 'пшеница', 'corn': 'кукуруза', 'barley': 'ячмень'}.get(cls, cls)
        print(f"   {class_ru}: {count} изображений")
    
    return image_paths, labels, visual_features_list

def create_data_augmentation():
    """Создание аугментации данных"""
    return {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

def train_model():
    """Обучение улучшенной модели"""
    print("🚀 ОБУЧЕНИЕ УЛУЧШЕННОЙ МОДЕЛИ КЛАССИФИКАЦИИ КУЛЬТУР")
    print("=" * 60)
    
    # Загружаем данные
    image_paths, labels, visual_features_list = load_analyzed_data()
    
    if not image_paths:
        return None
    
    # Проверяем наличие минимального количества данных
    if len(image_paths) < 10:
        print("⚠️ Недостаточно данных для обучения (минимум 10 изображений)")
        print("Генерируем дополнительные синтетические данные...")
        
        # Генерируем синтетические данные для недостающих классов
        image_paths, labels, visual_features_list = generate_synthetic_data(
            image_paths, labels, visual_features_list
        )
    
    # Разделяем данные
    if len(set(labels)) < 2:
        print("⚠️ Недостаточно разнообразия в классах. Добавляем базовые примеры...")
        # Добавляем базовые примеры для каждого класса
        image_paths, labels, visual_features_list = add_base_examples(
            image_paths, labels, visual_features_list
        )
    
    # Создаем разделение данных
    train_paths, val_paths, train_labels, val_labels, train_features, val_features = train_test_split(
        image_paths, labels, visual_features_list, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"📊 Обучающая выборка: {len(train_paths)} изображений")
    print(f"📊 Валидационная выборка: {len(val_paths)} изображений")
    
    # Создаем трансформации
    transforms_dict = create_data_augmentation()
    
    # Создаем датасеты
    train_dataset = CropPhotoDataset(
        train_paths, train_labels, train_features, 
        transform=transforms_dict['train'], use_features=True
    )
    
    val_dataset = CropPhotoDataset(
        val_paths, val_labels, val_features,
        transform=transforms_dict['val'], use_features=True
    )
    
    # Создаем загрузчики данных
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Создаем модель
    device = torch.device(settings.DEVICE)
    model = EnhancedCropClassifier(num_classes=3, use_visual_features=True)
    model.to(device)
    
    # Настройки обучения
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Обучение
    num_epochs = 20
    best_val_acc = 0.0
    
    train_losses = []
    val_accuracies = []
    
    print("\n🎯 Начинаем обучение...")
    
    for epoch in range(num_epochs):
        # Обучение
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, visual_features, labels in train_loader:
            images = images.to(device)
            visual_features = visual_features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, visual_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_acc = 100 * correct_train / total_train
        avg_train_loss = running_loss / len(train_loader)
        
        # Валидация
        model.eval()
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, visual_features, labels in val_loader:
                images = images.to(device)
                visual_features = visual_features.to(device)
                labels = labels.to(device)
                
                outputs = model(images, visual_features)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_acc = 100 * correct_val / total_val
        
        # Обновляем расписание обучения
        scheduler.step(1 - val_acc/100)
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)
        
        print(f"Эпоха [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Val Acc: {val_acc:.2f}%")
        
        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_save_path = "data/models/enhanced_crop_classifier.pth"
            Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'class_to_idx': train_dataset.class_to_idx
            }, model_save_path)
            
            print(f"✅ Сохранена лучшая модель с точностью {val_acc:.2f}%")
    
    print(f"\n🎉 Обучение завершено! Лучшая точность: {best_val_acc:.2f}%")
    
    # Построение графиков
    plot_training_history(train_losses, val_accuracies)
    
    return model, best_val_acc

def generate_synthetic_data(image_paths, labels, visual_features_list):
    """Генерация синтетических данных для увеличения датасета"""
    print("🔧 Генерация синтетических данных...")
    
    # Проверяем какие классы отсутствуют
    present_classes = set(labels)
    all_classes = {'wheat', 'corn', 'barley'}
    missing_classes = all_classes - present_classes
    
    synthetic_paths = []
    synthetic_labels = []
    synthetic_features = []
    
    for missing_class in missing_classes:
        # Создаем несколько синтетических примеров для каждого отсутствующего класса
        for i in range(5):
            # Генерируем синтетические визуальные признаки
            if missing_class == 'wheat':
                features = {
                    'green_ratio': np.random.uniform(0.1, 0.4),
                    'yellow_ratio': np.random.uniform(0.3, 0.7),
                    'vertical_lines': np.random.uniform(0.2, 0.4),
                    'edge_density': np.random.uniform(0.15, 0.25),
                    'aspect_ratio': np.random.uniform(0.7, 1.3)
                }
            elif missing_class == 'corn':
                features = {
                    'green_ratio': np.random.uniform(0.5, 0.8),
                    'yellow_ratio': np.random.uniform(0.1, 0.3),
                    'vertical_lines': np.random.uniform(0.4, 0.6),
                    'edge_density': np.random.uniform(0.1, 0.2),
                    'aspect_ratio': np.random.uniform(1.2, 1.8)
                }
            else:  # barley
                features = {
                    'green_ratio': np.random.uniform(0.3, 0.6),
                    'yellow_ratio': np.random.uniform(0.2, 0.4),
                    'vertical_lines': np.random.uniform(0.1, 0.3),
                    'edge_density': np.random.uniform(0.12, 0.22),
                    'aspect_ratio': np.random.uniform(0.8, 1.2)
                }
            
            synthetic_paths.append(f"synthetic_{missing_class}_{i}")
            synthetic_labels.append(missing_class)
            synthetic_features.append(features)
    
    print(f"✅ Добавлено {len(synthetic_paths)} синтетических примеров")
    
    return image_paths + synthetic_paths, labels + synthetic_labels, visual_features_list + synthetic_features

def add_base_examples(image_paths, labels, visual_features_list):
    """Добавление базовых примеров для обеспечения минимального разнообразия"""
    print("📝 Добавление базовых примеров...")
    
    # Базовые характеристики для каждого класса
    base_examples = {
        'wheat': {
            'green_ratio': 0.25,
            'yellow_ratio': 0.5,
            'vertical_lines': 0.3,
            'edge_density': 0.2,
            'aspect_ratio': 1.0
        },
        'corn': {
            'green_ratio': 0.65,
            'yellow_ratio': 0.2,
            'vertical_lines': 0.5,
            'edge_density': 0.15,
            'aspect_ratio': 1.5
        },
        'barley': {
            'green_ratio': 0.45,
            'yellow_ratio': 0.3,
            'vertical_lines': 0.2,
            'edge_density': 0.18,
            'aspect_ratio': 1.1
        }
    }
    
    present_classes = set(labels)
    all_classes = {'wheat', 'corn', 'barley'}
    
    for class_name in all_classes:
        if class_name not in present_classes:
            image_paths.append(f"base_example_{class_name}")
            labels.append(class_name)
            visual_features_list.append(base_examples[class_name])
    
    return image_paths, labels, visual_features_list

def plot_training_history(train_losses, val_accuracies):
    """Построение графиков обучения"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # График потерь
        ax1.plot(train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # График точности
        ax2.plot(val_accuracies)
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print("📊 График обучения сохранен в training_history.png")
        
    except Exception as e:
        print(f"⚠️ Не удалось построить графики: {e}")

if __name__ == "__main__":
    print("🌾 СИСТЕМА ОБУЧЕНИЯ МОДЕЛИ НА РЕАЛЬНЫХ ФОТОГРАФИЯХ")
    print("🎯 Использование проанализированных фото из папки photo/")
    print("=" * 65)
    
    try:
        model, best_accuracy = train_model()
        
        if model and best_accuracy:
            print(f"\n🎉 ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
            print(f"🏆 Лучшая точность: {best_accuracy:.2f}%")
            print(f"💾 Модель сохранена в: data/models/enhanced_crop_classifier.pth")
            print(f"📊 График обучения: training_history.png")
            print("\n✅ Модель готова к использованию!")
        else:
            print("❌ Обучение не удалось завершить")
            
    except Exception as e:
        print(f"💥 Ошибка во время обучения: {e}")
        import traceback
        traceback.print_exc() 