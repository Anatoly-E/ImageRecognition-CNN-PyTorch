import torch
import torch.nn as nn
import torchvision
from .config import Config

class ResNet18(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(ResNet18, self).__init__()
        
        # ИСПРАВЛЕННЫЙ КОД - без предупреждений
        self.resnet = torchvision.models.resnet18(weights=None)
        
        # Адаптируем под CIFAR-10 (32x32 вместо 224x224)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # Убираем maxpool
        self.resnet.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

def create_model(device=Config.DEVICE):
    """Создание и инициализация модели"""
    model = ResNet18().to(device)
    print(f"Модель создана и перемещена на {device}")
    print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")
    return model