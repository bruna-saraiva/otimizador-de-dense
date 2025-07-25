import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data import create_loaders, get_class_weights
from build_and_train import build_and_train
from config import BASE_DIR, RESULTS_DIR, num_classes_exp, device
# Adicione no início do run_holdout.py



# Hyperparameters
hype_space = {
    "compress_factor": 0.5,
    "dropout_rate": 0.2,
    "growth_rate": 16,
    "num_blocks": 3,
    "num_filters": 32,
    "num_layers_per_block": 2,
    "se_config": "apenas_H"
}

os.makedirs(RESULTS_DIR, exist_ok=True)
results = []

for split_num in range(1, 6):
    print(f"\n=== Processing split {split_num} ===")
    
    # Data loaders
    train_dir = os.path.join(BASE_DIR, f"split{split_num}", "train")
    val_dir = os.path.join(BASE_DIR, f"split{split_num}", "val")
    test_dir = os.path.join(BASE_DIR, f"split{split_num}", "test")
    
    train_loader, val_loader, test_loader = create_loaders(train_dir, val_dir, test_dir)
    class_weight = get_class_weights(train_loader)
    
    # Training
    result = build_and_train(hype_space, train_loader, val_loader, test_loader, class_weight)
    results.append(result)
    
    # Save model
    model_path = os.path.join(RESULTS_DIR, f"model_split_{split_num}.pth")
    torch.save(result['model'].state_dict(), model_path)

    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(RESULTS_DIR, f'cm_split_{split_num}.png'))
    plt.close()

# Calculate average metrics
avg_metrics = {
    'accuracy': np.mean([r['accuracy'] for r in results]),
    'precision': np.mean([r['report']['weighted avg']['precision'] for r in results]),
    'recall': np.mean([r['report']['weighted avg']['recall'] for r in results]),
    'f1': np.mean([r['report']['weighted avg']['f1-score'] for r in results]),
    'std_accuracy': np.std([r['accuracy'] for r in results])
}

# Save results
# Por este novo código:
# 1. Primeiro salve os modelos separadamente
for split_num, result in enumerate(results, 1):
    model_path = os.path.join(RESULTS_DIR, f"model_split_{split_num}.pth")
    torch.save(result['model'].state_dict(), model_path)

# 2. Prepare dados serializáveis para o JSON
serializable_results = []
for split_num, result in enumerate(results, 1):
    serializable_results.append({
        'split_number': split_num,
        'history': result['history'],
        'accuracy': result['accuracy'],
        'report': result['report'],
        'confusion_matrix': result['confusion_matrix'],
        'training_time': result['training_time'],
        'model_path': f"model_split_{split_num}.pth"
    })

# 3. Salve o JSON com os dados serializáveis
with open(os.path.join(RESULTS_DIR, 'holdout_results.json'), 'w') as f:
    json.dump({
        'hyperparameters': hype_space,
        'individual_results': serializable_results,
        'average_metrics': avg_metrics
    }, f, indent=4)

print("\n=== Final Results ===")
print(f"Average Accuracy: {avg_metrics['accuracy']:.4f} ± {avg_metrics['std_accuracy']:.4f}")
print(f"Average Precision: {avg_metrics['precision']:.4f}")
print(f"Average Recall: {avg_metrics['recall']:.4f}")
print(f"Average F1-Score: {avg_metrics['f1']:.4f}")
print(f"\nResults saved in {RESULTS_DIR}")