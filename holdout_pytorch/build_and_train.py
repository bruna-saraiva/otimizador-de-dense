import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from customized_model import get_model
from data import create_loaders, get_class_weights
from config import img_height, img_width, batch_size, batch_size_val, num_classes_exp, RESULTS_DIR, epochs, device  
import os

def build_and_train(hype_space, train_loader, val_loader, test_loader, class_weight):
    model = get_model(
        input_shape=(1, img_height, img_width),
        num_blocks=hype_space['num_blocks'],
        num_layers_per_block=hype_space['num_layers_per_block'],
        growth_rate=hype_space['growth_rate'],
        dropout_rate=hype_space['dropout_rate'],
        compress_factor=hype_space['compress_factor'],
        num_filters=hype_space['num_filters'],
        num_classes=num_classes_exp,
        se_config=hype_space['se_config']
    ).to(device)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_model_path = os.path.join(RESULTS_DIR, 'best_model.pth')

    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    start_time = time.time()
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_correct = 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == targets).sum().item()
        
        # Validation
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                val_correct += (outputs.argmax(1) == targets).sum().item()
        
        # Metrics
        train_loss /= len(train_loader)
        train_acc = train_correct / len(train_loader.dataset)
        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"\nNew best model saved to {best_model_path}")
    # Evaluation

    if os.path.exists(best_model_path):
        print("\nLoading best model for evaluation...")
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        model.to(device)
    else:
        print("\nNo best model found, using final model weights for evaluation")
    
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
    
    acc = accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, output_dict=True)
    cm = confusion_matrix(all_targets, all_preds)
    
    return {
        'model': model,
        'history': history,
        'accuracy': acc,
        'report': report,
        'confusion_matrix': cm.tolist(),
        'training_time': time.time() - start_time
    }