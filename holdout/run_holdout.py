from data import create_generators, get_class_weights
from build_and_train import build_and_train
from config import BASE_DIR, img_height, img_width, batch_size, batch_size_val, RESULTS_DIR, num_classes_exp
import os
import json
import numpy as np


# Hiperparâmetros fixos
hype_space = {
    "compress_factor": 0.5,
    "dropout_rate": 0.2612908742849015,
    "growth_rate": 16,
    "num_blocks": 3,
    "num_filters": 32,
    "num_layers_per_block": 2,
    "se_config": "apenas_H"
}

os.makedirs(RESULTS_DIR, exist_ok=True)

results = []
metrics_summary = {
    'accuracy': [],
    'precision': {i: [] for i in range(num_classes_exp)},
    'recall': {i: [] for i in range(num_classes_exp)},
    'f1-score': {i: [] for i in range(num_classes_exp)},
    'support': {i: [] for i in range(num_classes_exp)}
}

for split_num in range(1, 6):
    print(f"\n=== Processando split {split_num} ===")
    
    # Diretórios
    train_dir = os.path.join(BASE_DIR, f"split{split_num}", "train")
    val_dir = os.path.join(BASE_DIR, f"split{split_num}", "val")
    test_dir = os.path.join(BASE_DIR, f"split{split_num}", "test")
    
    # Generators
    train_gen, val_gen, test_gen = create_generators(
        train_dir, val_dir, test_dir,
        img_height, img_width,
        batch_size, batch_size_val
    )
    
    # Class weights
    class_weight = get_class_weights(train_gen)
    
    # Treinamento
    result = build_and_train(hype_space, train_gen, val_gen, test_gen, class_weight)
    
    # Armazena resultados brutos
    results.append({
        'split': split_num,
        **result
    })
    
    # Coleta métricas para sumário
    metrics_summary['accuracy'].append(result['accuracy'])
    
    for class_id in range(num_classes_exp):
        metrics_summary['precision'][class_id].append(result['report'][str(class_id)]['precision'])
        metrics_summary['recall'][class_id].append(result['report'][str(class_id)]['recall'])
        metrics_summary['f1-score'][class_id].append(result['report'][str(class_id)]['f1-score'])
        metrics_summary['support'][class_id].append(result['report'][str(class_id)]['support'])

    # Salva modelo
    model_path = os.path.join(RESULTS_DIR, f"model_split_{split_num}.keras")
    result['model'].save(model_path)

# Calcula estatísticas
final_report = {
    'overall': {
        'accuracy': {
            'mean': np.mean(metrics_summary['accuracy']),
            'std': np.std(metrics_summary['accuracy'])
        }
    },
    'per_class': {}
}

for class_id in range(num_classes_exp):
    final_report['per_class'][f'class_{class_id}'] = {
        'precision': {
            'mean': np.mean(metrics_summary['precision'][class_id]),
            'std': np.std(metrics_summary['precision'][class_id])
        },
        'recall': {
            'mean': np.mean(metrics_summary['recall'][class_id]),
            'std': np.std(metrics_summary['recall'][class_id])
        },
        'f1-score': {
            'mean': np.mean(metrics_summary['f1-score'][class_id]),
            'std': np.std(metrics_summary['f1-score'][class_id])
        },
        'support': int(np.mean(metrics_summary['support'][class_id]))  # Support é constante
    }

# Salva tudo
output = {
    'individual_results': results,
    'aggregated_metrics': final_report
}

with open(os.path.join(RESULTS_DIR, "holdout_results.json"), "w") as f:
    json.dump(output, f, indent=4)

print("\n=== Relatório Final ===")
print(f"Acurácia: {final_report['overall']['accuracy']['mean']:.4f} ± {final_report['overall']['accuracy']['std']:.4f}")
for class_id in range(num_classes_exp):
    print(f"\nClasse {class_id}:")
    print(f"  Precision: {final_report['per_class'][f'class_{class_id}']['precision']['mean']:.4f} ± {final_report['per_class'][f'class_{class_id}']['precision']['std']:.4f}")
    print(f"  Recall:    {final_report['per_class'][f'class_{class_id}']['recall']['mean']:.4f} ± {final_report['per_class'][f'class_{class_id}']['recall']['std']:.4f}")
    print(f"  F1-Score:  {final_report['per_class'][f'class_{class_id}']['f1-score']['mean']:.4f} ± {final_report['per_class'][f'class_{class_id}']['f1-score']['std']:.4f}")

print("\nHold-out concluído! Resultados salvos em:", RESULTS_DIR)