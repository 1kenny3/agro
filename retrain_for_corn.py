#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌽 ПЕРЕОБУЧЕНИЕ МОДЕЛИ ДЛЯ ПРАВИЛЬНОГО РАСПОЗНАВАНИЯ КУКУРУЗЫ
================================================================
Специальное переобучение с правильной разметкой images.jpeg как кукуруза
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import timm

class CornDataset(Dataset):
    def __init__(self, image_paths, labels, visual_features_list, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.visual_features_list = visual_features_list
        self.transform = transform
        
        # Создаем маппинг классов
        self.class_to_idx = {'wheat': 0, 'corn': 1, 'barley': 2}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Загружаем изображение
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Получаем метку
        label = self.class_to_idx[self.labels[idx]]
        
        # Получаем визуальные признаки
        visual_features = self.visual_features_list[idx]
        visual_tensor = torch.tensor([
            visual_features.get('green_ratio', 0),
            visual_features.get('yellow_ratio', 0),
            visual_features.get('vertical_lines', 0),
            visual_features.get('edge_density', 0),
            visual_features.get('aspect_ratio', 1)
        ], dtype=torch.float32)
        
        return image, visual_tensor, label

class EnhancedCropClassifier(nn.Module):
    def __init__(self, num_classes=3, model_name="efficientnet_b0", use_visual_features=True):
        super().__init__()
        
        self.use_visual_features = use_visual_features
        
        # Используем timm для извлечения признаков
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
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

def create_corrected_dataset():
    """Создаем датасет с исправленной разметкой"""
    print("📊 СОЗДАНИЕ ИСПРАВЛЕННОГО ДАТАСЕТА")
    print("=" * 50)
    
    # Загружаем оригинальные результаты анализа
    with open("photo_analysis_results.json", 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    image_paths = []
    labels = []
    visual_features_list = []
    
    for result in results:
        # Пропускаем файлы с ошибками
        if result.get('visual_features', {}).get('error'):
            continue
        
        file_path = result['file_info']['path']
        filename = Path(file_path).name.lower()
        
        # ИСПРАВЛЕННАЯ РАЗМЕТКА
        if 'images.jpeg' in filename:
            # Пользователь указал, что это кукуруза
            correct_label = 'corn'
            print(f"🌽 {filename} -> КУКУРУЗА (исправлено)")
        elif '119252' in filename:
            correct_label = 'corn'
            print(f"🌽 {filename} -> кукуруза")
        elif any(word in filename for word in ['wheat', 'пшеница', 'пшениц']):
            correct_label = 'wheat'
            print(f"🌾 {filename} -> пшеница")
        elif filename.startswith('dji_'):
            # Анализируем DJI фото по признакам
            visual_features = result['visual_features']
            yellow_ratio = visual_features.get('yellow_ratio', 0)
            green_ratio = visual_features.get('green_ratio', 0)
            
            if yellow_ratio > 0.3 and green_ratio < 0.1:
                if any(num in filename for num in ['0046', '0048', '0031']):
                    correct_label = 'wheat'
                    print(f"🌾 {filename} -> пшеница")
                else:
                    correct_label = 'barley'
                    print(f"🌿 {filename} -> ячмень")
            else:
                correct_label = 'barley'
                print(f"🌿 {filename} -> ячмень")
        else:
            correct_label = 'barley'
            print(f"🌿 {filename} -> ячмень (по умолчанию)")
        
        image_paths.append(file_path)
        labels.append(correct_label)
        visual_features_list.append(result['visual_features'])
    
    # Статистика
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print(f"\n📊 Исправленный датасет: {len(image_paths)} изображений")
    print("Распределение по классам:")
    for cls, count in class_counts.items():
        class_ru = {'wheat': 'пшеница', 'corn': 'кукуруза', 'barley': 'ячмень'}[cls]
        print(f"   {class_ru}: {count} изображений")
    
    return image_paths, labels, visual_features_list

def train_corrected_model():
    """Обучение модели с исправленными данными"""
    print("\n🚀 ПЕРЕОБУЧЕНИЕ МОДЕЛИ С ИСПРАВЛЕННЫМИ ДАННЫМИ")
    print("=" * 60)
    
    # Создаем исправленный датасет
    image_paths, labels, visual_features_list = create_corrected_dataset()
    
    if len(image_paths) < 5:
        print("❌ Недостаточно данных для обучения!")
        return
    
    # Разделяем данные (если достаточно для валидации)
    if len(set(labels)) >= 2 and len(image_paths) >= 8:
        train_paths, val_paths, train_labels, val_labels, train_features, val_features = train_test_split(
            image_paths, labels, visual_features_list, test_size=0.2, random_state=42, stratify=labels
        )
    else:
        # Используем все данные для обучения
        train_paths, train_labels, train_features = image_paths, labels, visual_features_list
        val_paths, val_labels, val_features = image_paths[:2], labels[:2], visual_features_list[:2]
    
    print(f"📊 Обучающая выборка: {len(train_paths)} изображений")
    print(f"📊 Валидационная выборка: {len(val_paths)} изображений")
    
    # Создаем трансформации
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Создаем датасеты
    train_dataset = CornDataset(train_paths, train_labels, train_features, transform=train_transform)
    val_dataset = CornDataset(val_paths, val_labels, val_features, transform=val_transform)
    
    # Создаем загрузчики данных
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Создаем модель
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Устройство: {device}")
    
    model = EnhancedCropClassifier(num_classes=3, use_visual_features=True)
    model.to(device)
    
    # Настройки обучения
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Обучение
    num_epochs = 30
    best_acc = 0.0
    
    print("\n🎯 Начинаем обучение...")
    
    for epoch in range(num_epochs):
        # Обучение
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, visual_features, labels in train_loader:
            images = images.to(device)
            visual_features = visual_features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, visual_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Валидация
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, visual_features, labels in val_loader:
                images = images.to(device)
                visual_features = visual_features.to(device)
                labels = labels.to(device)
                
                outputs = model(images, visual_features)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Эпоха [{epoch+1}/{num_epochs}] - Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Сохраняем лучшую модель
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"✅ Сохранена лучшая модель с точностью {best_acc:.2f}%")
            
            # Создаем папку если не существует
            Path("data/models").mkdir(parents=True, exist_ok=True)
            
            # Сохраняем модель
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_accuracy': best_acc,
                'class_to_idx': train_dataset.class_to_idx,
                'epoch': epoch
            }, "data/models/enhanced_crop_classifier.pth")
    
    print(f"\n🎉 Обучение завершено! Лучшая точность: {best_acc:.2f}%")
    print(f"💾 Модель сохранена в: data/models/enhanced_crop_classifier.pth")

if __name__ == "__main__":
    train_corrected_model() 