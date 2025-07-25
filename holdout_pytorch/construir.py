import torch
import os
from customized_model import get_model  # Importe sua função que cria a arquitetura
from config import img_height, img_width, num_classes_exp, device

# Configurações
WEIGHTS_PATH = "/home/bruna/Pibic-2024/Aulas Visao/otimizador de dense/holdout_pytorch/resultados/best_model.pth"  # Substitua pelo caminho real
SAVE_DIR = "/home/bruna/Pibic-2024/Aulas Visao/otimizador de dense/holdout_pytorch"      # Pasta onde será salvo
MODEL_NAME = "modelo_completo.pth"                     # Nome do arquivo de saída

# Hiperparâmetros usados originalmente (DEVE ser idêntico ao treinamento)
hype_space = {
    "compress_factor": 0.5,
    "dropout_rate": 0.261,
    "growth_rate": 16,
    "num_blocks": 3,
    "num_filters": 32,
    "num_layers_per_block": 2,
    "se_config": "apenas_H"
}

def load_and_save_full_model():
    # 1. Verificar se a pasta de destino existe
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 2. Recriar a arquitetura do modelo
    print("Recriando a arquitetura do modelo...")
    model = get_model(
        input_shape=(1, img_height, img_width),
        num_classes=num_classes_exp,
        **hype_space
    ).to(device)
    
    # 3. Carregar os pesos salvos
    print(f"Carregando pesos de {WEIGHTS_PATH}...")
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 4. Salvar o modelo completo
    save_path = os.path.join(SAVE_DIR, MODEL_NAME)
    torch.save(model, save_path)
    print(f"Modelo completo salvo em: {save_path}")
    
    # 5. Verificação (opcional)
    print("\nVerificando o modelo salvo...")
    loaded_model = torch.load(save_path, map_location=device)
    print("Arquitetura do modelo carregado:")
    print(loaded_model)
    
    # Teste com input dummy
    dummy_input = torch.randn(1, 1, img_height, img_width).to(device)
    with torch.no_grad():
        output = loaded_model(dummy_input)
        print("\nSaída do modelo recarregado (shape):", output.shape)

if __name__ == "__main__":
    load_and_save_full_model()