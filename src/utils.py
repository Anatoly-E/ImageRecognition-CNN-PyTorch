import matplotlib.pyplot as plt
import torch
import pandas as pd
import os
from .config import Config

class Visualizer:
    def __init__(self):
        self.config = Config
    
    def plot_training_history(self, history_aug, history_basic, save_path=None):
        """Визуализация истории обучения"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history_aug['train_accs'], label='С аугментацией (train)', linewidth=2)
        axes[0, 0].plot(history_aug['val_accs'], label='С аугментацией (val)', linewidth=2)
        axes[0, 0].set_title('Accuracy - С аугментацией', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(history_basic['train_accs'], label='Без аугментации (train)', linewidth=2)
        axes[0, 1].plot(history_basic['val_accs'], label='Без аугментации (val)', linewidth=2)
        axes[0, 1].set_title('Accuracy - Без аугментации', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss
        axes[1, 0].plot(history_aug['train_losses'], label='С аугментацией (train)', linewidth=2)
        axes[1, 0].plot(history_aug['val_losses'], label='С аугментацией (val)', linewidth=2)
        axes[1, 0].set_title('Loss - С аугментацией', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(history_basic['train_losses'], label='Без аугментации (train)', linewidth=2)
        axes[1, 1].plot(history_basic['val_losses'], label='Без аугментации (val)', linewidth=2)
        axes[1, 1].set_title('Loss - Без аугментации', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
            print(f"График сохранен: {os.path.join(save_path, 'training_history.png')}")
        
        plt.show()
    
    def plot_comparison(self, metrics_aug, metrics_basic, save_path=None):
        """Визуализация сравнения моделей"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        models = ['С аугментацией', 'Без аугментации']
        colors = ['#2E8B57', '#DC143C']  # Зеленый и красный
        
        # Accuracy
        accuracies = [metrics_aug['test_accuracy'], metrics_basic['test_accuracy']]
        bars = axes[0].bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_title('Сравнение точности на тестовом наборе', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Accuracy (%)')
        for bar, acc in zip(bars, accuracies):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # Loss
        losses = [metrics_aug['test_loss'], metrics_basic['test_loss']]
        bars = axes[1].bar(models, losses, color=colors, alpha=0.7, edgecolor='black')
        axes[1].set_title('Сравнение потерь на тестовом наборе', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Loss')
        for bar, loss in zip(bars, losses):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                        f'{loss:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Validation Accuracy
        val_accs = [metrics_aug['best_val_accuracy'] * 100, metrics_basic['best_val_accuracy'] * 100]
        bars = axes[2].bar(models, val_accs, color=colors, alpha=0.7, edgecolor='black')
        axes[2].set_title('Лучшая точность на валидации', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Accuracy (%)')
        for bar, acc in zip(bars, val_accs):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'model_comparison.png'), dpi=300, bbox_inches='tight')
            print(f"График сравнения сохранен: {os.path.join(save_path, 'model_comparison.png')}")
        
        plt.show()

class MetricsCalculator:
    def __init__(self):
        self.config = Config
    
    def calculate_class_wise_accuracy(self, model, test_loader, device):
        """Вычисление точности по классам"""
        model.eval()
        class_correct = list(0. for _ in range(self.config.NUM_CLASSES))
        class_total = list(0. for _ in range(self.config.NUM_CLASSES))
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        class_accuracies = {}
        print("\nТочность по классам:")
        for i in range(self.config.NUM_CLASSES):
            if class_total[i] > 0:
                accuracy = 100 * class_correct[i] / class_total[i]
                class_accuracies[self.config.CLASSES[i]] = accuracy
                print(f'  {self.config.CLASSES[i]:10s}: {accuracy:.2f}%')
        
        return class_accuracies
    
    def save_metrics_to_csv(self, metrics_aug, metrics_basic, save_path):
        """Сохранение метрик в CSV"""
        comparison_data = {
            'Метрика': ['Test Accuracy', 'Test Loss', 'Best Val Accuracy'],
            'С аугментацией': [
                f"{metrics_aug['test_accuracy']:.2f}%",
                f"{metrics_aug['test_loss']:.4f}",
                f"{metrics_aug['best_val_accuracy']:.4f}"
            ],
            'Без аугментации': [
                f"{metrics_basic['test_accuracy']:.2f}%", 
                f"{metrics_basic['test_loss']:.4f}",
                f"{metrics_basic['best_val_accuracy']:.4f}"
            ],
            'Разница': [
                f"{metrics_aug['test_accuracy'] - metrics_basic['test_accuracy']:+.2f}%",
                f"{metrics_aug['test_loss'] - metrics_basic['test_loss']:+.4f}",
                f"{metrics_aug['best_val_accuracy'] - metrics_basic['best_val_accuracy']:+.4f}"
            ]
        }
        
        df = pd.DataFrame(comparison_data)
        csv_path = os.path.join(save_path, 'metrics_comparison.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Метрики сохранены: {csv_path}")
        
        return df