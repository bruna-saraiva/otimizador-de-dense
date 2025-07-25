import os
import shutil
from sklearn.model_selection import train_test_split

# Definindo os caminhos
base_dir = 'chest_pneumonia'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')  # Pasta de validação que será criada
test_dir = os.path.join(base_dir, 'test')

# Criar pasta de validação se não existir
os.makedirs(val_dir, exist_ok=True)

# Proporções desejadas
total_images = 5854
val_size = 585  # 10% do total
test_size = 627  # já existente

# Para cada classe (0, 1, 2)
for class_id in ['0', '1', '2']:
    # Caminhos das pastas
    class_train_dir = os.path.join(train_dir, class_id)
    class_val_dir = os.path.join(val_dir, class_id)
    
    # Criar pasta de validação para a classe
    os.makedirs(class_val_dir, exist_ok=True)
    
    # Listar todas as imagens da classe
    images = os.listdir(class_train_dir)
    
    # Calcular quantas imagens mover para validação
    if class_id == '0':
        # 1341 no treino original, queremos ~1056 no novo treino (1341 - 285)
        n_val = 285
    elif class_id == '1':
        # 2538 no treino original, queremos ~2284 no novo treino (2538 - 254)
        n_val = 254
    elif class_id == '2':
        # 1345 no treino original, queremos ~1210 no novo treino (1345 - 135)
        n_val = 135
    
    # Selecionar aleatoriamente as imagens para validação
    val_images = set(os.listdir(class_train_dir)[:n_val])
    
    # Mover as imagens selecionadas
    for image in val_images:
        src = os.path.join(class_train_dir, image)
        dst = os.path.join(class_val_dir, image)
        shutil.move(src, dst)

print("Validação criada com sucesso!")