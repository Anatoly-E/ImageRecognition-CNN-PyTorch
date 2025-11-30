import sys
import os

# Добавляем папку src в путь для импорта
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config
from src.data_loader import DataProcessor
from src.model import create_model
from src.train import ModelTrainer
from src.utils import Visualizer, MetricsCalculator

def main():
    """Главная функция проекта"""
    print("🚀 ЗАПУСК ПРОЕКТА ПО КЛАССИФИКАЦИИ CIFAR-10")
    print("=" * 60)
    
    # Показываем конфигурацию
    Config.print_config()
    
    # Создаем экземпляры классов
    data_processor = DataProcessor()
    trainer = ModelTrainer()
    visualizer = Visualizer()
    metrics_calculator = MetricsCalculator()
    
    # Шаг 1: Подготовка данных
    print("\n📂 ПОДГОТОВКА ДАННЫХ...")
    dataloaders = data_processor.prepare_dataloaders()
    
    # Шаг 2: Обучение модели с аугментацией
    print("\n" + "="*60)
    print("🎯 ОБУЧЕНИЕ МОДЕЛИ С АУГМЕНТАЦИЕЙ ДАННЫХ")
    print("="*60)
    
    model_aug = create_model()
    history_aug = trainer.train_model(
        model_aug, 
        dataloaders['augmented']['train'], 
        dataloaders['augmented']['val'], 
        model_name='with_augmentation'
    )
    
    # Шаг 3: Обучение модели без аугментации
    print("\n" + "="*60)
    print("🎯 ОБУЧЕНИЕ МОДЕЛИ БЕЗ АУГМЕНТАЦИИ ДАННЫХ")
    print("="*60)
    
    model_basic = create_model()
    history_basic = trainer.train_model(
        model_basic,
        dataloaders['basic']['train'],
        dataloaders['basic']['val'],
        model_name='without_augmentation'
    )
    
    # Шаг 4: Тестирование моделей
    print("\n" + "="*60)
    print("🧪 ТЕСТИРОВАНИЕ МОДЕЛЕЙ НА TEST SET")
    print("="*60)
    
    # Загружаем лучшие веса
    model_aug.load_state_dict(torch.load(f'{Config.MODEL_SAVE_PATH}/best_with_augmentation.pth'))
    model_basic.load_state_dict(torch.load(f'{Config.MODEL_SAVE_PATH}/best_without_augmentation.pth'))
    
    # Оценка моделей
    acc_aug, loss_aug = trainer.evaluate_model(model_aug, dataloaders['test'], "С аугментацией")
    acc_basic, loss_basic = trainer.evaluate_model(model_basic, dataloaders['test'], "Без аугментации")
    
    # Подготавливаем метрики для визуализации
    metrics_aug = {
        'test_accuracy': acc_aug,
        'test_loss': loss_aug,
        'best_val_accuracy': history_aug['best_val_accuracy']
    }
    
    metrics_basic = {
        'test_accuracy': acc_basic,
        'test_loss': loss_basic,
        'best_val_accuracy': history_basic['best_val_accuracy']
    }
    
    # Шаг 5: Визуализация результатов
    print("\n" + "="*60)
    print("📊 ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("="*60)
    
    visualizer.plot_training_history(history_aug, history_basic, Config.PLOT_SAVE_PATH)
    visualizer.plot_comparison(metrics_aug, metrics_basic, Config.PLOT_SAVE_PATH)
    
    # Шаг 6: Детальный анализ
    print("\n" + "="*60)
    print("🔍 ДЕТАЛЬНЫЙ АНАЛИЗ")
    print("="*60)
    
    print("\nМОДЕЛЬ С АУГМЕНТАЦИЕЙ:")
    class_accuracies_aug = metrics_calculator.calculate_class_wise_accuracy(
        model_aug, dataloaders['test'], Config.DEVICE
    )
    
    print("\nМОДЕЛЬ БЕЗ АУГМЕНТАЦИИ:")
    class_accuracies_basic = metrics_calculator.calculate_class_wise_accuracy(
        model_basic, dataloaders['test'], Config.DEVICE
    )
    
    # Шаг 7: Сохранение метрик
    print("\n" + "="*60)
    print("💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*60)
    
    df_metrics = metrics_calculator.save_metrics_to_csv(
        metrics_aug, metrics_basic, Config.METRICS_SAVE_PATH
    )
    print("\nСохраненные метрики:")
    print(df_metrics)
    
    # Шаг 8: Выводы
    print("\n" + "="*60)
    print("🎯 ВЫВОДЫ И ЗАКЛЮЧЕНИЕ")
    print("="*60)
    
    improvement = acc_aug - acc_basic
    if improvement > 0:
        print(f"✅ АУГМЕНТАЦИЯ ДАННЫХ ПОЛОЖИТЕЛЬНО ПОВЛИЯЛА НА МОДЕЛЬ")
        print(f"   Улучшение точности: +{improvement:.2f}%")
        print(f"   Относительное улучшение: +{(improvement/acc_basic*100):.2f}%")
    else:
        print(f"⚠️ АУГМЕНТАЦИЯ ДАННЫХ НЕ ДАЛА УЛУЧШЕНИЯ")
        print(f"   Изменение точности: {improvement:.2f}%")
    
    print("\n📈 Ключевые наблюдения:")
    print("1. Аугментация данных помогает бороться с переобучением")
    print("2. Модель с аугментацией обычно показывает более плавные кривые обучения")
    print("3. Разница в производительности особенно заметна на небольших датасетах")
    print("4. Правильная аугментация улучшает обобщающую способность модели")
    
    print(f"\n🎉 ПРОЕКТ УСПЕШНО ЗАВЕРШЕН!")
    print(f"📁 Результаты сохранены в папке: {Config.PLOT_SAVE_PATH}")

if __name__ == '__main__':
    import torch
    main()