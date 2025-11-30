import torch
import torchvision

class Config:
    # Пути
    DATA_PATH = './data'
    MODEL_SAVE_PATH = './outputs/models'
    PLOT_SAVE_PATH = './outputs/plots'
    METRICS_SAVE_PATH = './outputs/metrics'
    
    # Параметры данных
    BATCH_SIZE = 128
    NUM_WORKERS = 0  # Для избежания проблем с multiprocessing
    TRAIN_VAL_SPLIT = 0.8
    
    # Параметры модели
    NUM_CLASSES = 10
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 15
    
    # Аугментации
    RANDOM_CROP_PADDING = 4
    RANDOM_HORIZONTAL_FLIP_PROB = 0.5
    RANDOM_ROTATION_DEGREES = 10
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CIFAR-10 classes
    CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    @classmethod
    def print_config(cls):
        print("=" * 50)
        print("КОНФИГУРАЦИЯ ПРОЕКТА")
        print("=" * 50)
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                print(f"{key}: {value}")
        print("=" * 50)