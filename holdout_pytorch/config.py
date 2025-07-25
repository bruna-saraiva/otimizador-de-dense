# Training config
import torch
img_height = 180
img_width = 180
batch_size = 8
batch_size_val = 1
epochs = 20
num_classes_exp = 3
RESULTS_DIR = "/home/bruna/Pibic-2024/Aulas Visao/otimizador de dense/holdout_pytorch/resultados"
BASE_DIR = "/home/bruna/Pibic-2024/Aulas Visao/otimizador de dense/database"
device = "cuda" if torch.cuda.is_available() else "cpu"