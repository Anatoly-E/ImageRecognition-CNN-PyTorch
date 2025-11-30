import os
import shutil

def create_project_structure():
    folders = [
        'src',
        'outputs/models',
        'outputs/plots', 
        'outputs/metrics',
        'data/cifar-10'
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Создана папка: {folder}")
    
    # Создаем пустые __init__.py файлы
    with open('src/__init__.py', 'w') as f:
        pass
        
    print("Структура проекта создана!")

if __name__ == '__main__':
    create_project_structure()