import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
from .config import Config

class ModelTrainer:
    def __init__(self, device=Config.DEVICE):
        self.config = Config
        self.device = device
    
    def train_model(self, model, train_loader, val_loader, model_name='model'):
        """Обучение модели"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE, 
                             weight_decay=self.config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        best_acc = 0.0
        start_time = time.time()
        
        print(f"\n🎯 Начало обучения модели: {model_name}")
        print("=" * 60)
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f'\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}')
            print('-' * 50)
            
            # Training phase
            model.train()
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in tqdm(train_loader, desc='Training', leave=False):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)
            
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc.cpu().numpy())
            
            # Validation phase
            model.eval()
            running_loss = 0.0
            running_corrects = 0
            
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc='Validation', leave=False):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            
            epoch_val_loss = running_loss / len(val_loader.dataset)
            epoch_val_acc = running_corrects.double() / len(val_loader.dataset)
            
            val_losses.append(epoch_val_loss)
            val_accs.append(epoch_val_acc.cpu().numpy())
            
            scheduler.step()
            
            print(f'✅ Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'📊 Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')
            
            # Сохраняем лучшую модель
            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                model_path = os.path.join(self.config.MODEL_SAVE_PATH, f'best_{model_name}.pth')
                torch.save(model.state_dict(), model_path)
                print(f'💾 Сохранена лучшая модель: {model_path}')
        
        time_elapsed = time.time() - start_time
        print(f'\n🏁 Обучение завершено за {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'🎯 Лучшая точность на валидации: {best_acc:.4f}')
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_val_accuracy': best_acc
        }
    
    def evaluate_model(self, model, test_loader, model_name="model"):
        """Оценка модели на тестовом наборе"""
        model.eval()
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        test_loss = 0
        
        print(f"\n🧪 Тестирование модели: {model_name}")
        print("-" * 40)
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Testing', leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = test_loss / total
        
        print(f'📊 Test Loss: {avg_loss:.4f}')
        print(f'🎯 Test Accuracy: {accuracy:.2f}%')
        print(f'✅ Correct/Total: {correct}/{total}')
        
        return accuracy, avg_loss