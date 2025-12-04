import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from .config import Config

class DataProcessor:
    def __init__(self):
        self.config = Config
        self.channel_means = None
        self.channel_stds = None
        
    def compute_dataset_stats(self):
        """Вычисление статистики датасета для нормализации"""
        transform = transforms.Compose([transforms.ToTensor()])
        temp_set = torchvision.datasets.CIFAR10(
            root=self.config.DATA_PATH, train=True, download=True, transform=transform
        )
        
        loader = DataLoader(temp_set, batch_size=self.config.BATCH_SIZE, num_workers=0)
        
        mean = 0.0
        std = 0.0
        for images, _ in loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)   # Среднее по каналам
            std += images.std(2).sum(0)     # STD по каналам
        
        mean /= len(temp_set)               # Нормализуем по всему датасету
        std /= len(temp_set)
        
        self.channel_means = mean
        self.channel_stds = std
        
        print(f"Вычисленные средние: {self.channel_means.tolist()}")
        print(f"Вычисленные std: {self.channel_stds.tolist()}")
        
        return mean, std
    
    def get_transforms(self):
        """Получение преобразований с аугментацией и без"""
        if self.channel_means is None:
            self.compute_dataset_stats()
        
        # Без аугментации
        transform_basic = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.channel_means, self.channel_stds)
        ])
        
        # С аугментацией
        transform_augmented = transforms.Compose([
            transforms.RandomCrop(32, padding=self.config.RANDOM_CROP_PADDING),
            transforms.RandomHorizontalFlip(p=self.config.RANDOM_HORIZONTAL_FLIP_PROB),
            transforms.RandomRotation(self.config.RANDOM_ROTATION_DEGREES),
            transforms.ToTensor(),
            transforms.Normalize(self.channel_means, self.channel_stds)
        ])
        
        return transform_basic, transform_augmented
    
    def prepare_dataloaders(self):
        """Подготовка всех DataLoader'ов"""
        transform_basic, transform_augmented = self.get_transforms()
        
        # Train set с аугментацией
        trainset_augmented = torchvision.datasets.CIFAR10(
            root=self.config.DATA_PATH, train=True, download=True, 
            transform=transform_augmented
        )
        
        # Train set без аугментации
        trainset_basic = torchvision.datasets.CIFAR10(
            root=self.config.DATA_PATH, train=True, download=True, 
            transform=transform_basic
        )
        
        # Test set (всегда без аугментации)
        testset = torchvision.datasets.CIFAR10(
            root=self.config.DATA_PATH, train=False, download=True, 
            transform=transform_basic
        )
        
        # Разделение train на train/validation
        train_size = int(self.config.TRAIN_VAL_SPLIT * len(trainset_augmented))
        val_size = len(trainset_augmented) - train_size
        
        # Для аугментированного набора
        train_augmented, val_augmented = random_split(trainset_augmented, [train_size, val_size])
        
        # Для базового набора
        train_basic, val_basic = random_split(trainset_basic, [train_size, val_size])
        
        # DataLoader'ы
        train_loader_augmented = DataLoader(
            train_augmented, batch_size=self.config.BATCH_SIZE, 
            shuffle=True, num_workers=self.config.NUM_WORKERS
        )
        val_loader_augmented = DataLoader(
            val_augmented, batch_size=self.config.BATCH_SIZE, 
            shuffle=False, num_workers=self.config.NUM_WORKERS
        )
        
        train_loader_basic = DataLoader(
            train_basic, batch_size=self.config.BATCH_SIZE, 
            shuffle=True, num_workers=self.config.NUM_WORKERS
        )
        val_loader_basic = DataLoader(
            val_basic, batch_size=self.config.BATCH_SIZE, 
            shuffle=False, num_workers=self.config.NUM_WORKERS
        )
        
        test_loader = DataLoader(
            testset, batch_size=self.config.BATCH_SIZE, 
            shuffle=False, num_workers=self.config.NUM_WORKERS
        )
        
        print(f"Размеры наборов:")
        print(f"  Train: {train_size}, Validation: {val_size}, Test: {len(testset)}")
        
        return {
            'augmented': {'train': train_loader_augmented, 'val': val_loader_augmented},
            'basic': {'train': train_loader_basic, 'val': val_loader_basic},
            'test': test_loader
        }