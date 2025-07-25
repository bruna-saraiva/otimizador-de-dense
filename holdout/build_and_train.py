from customized_model import get_model
from config import epochs, RESULTS_DIR, img_height, img_width, num_classes_exp 
import tensorflow as tf
import numpy as np
import os
import time
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def build_and_train(hype_space, train_gen, val_gen, test_gen, class_weight):
    # Construção do modelo
    model = get_model(
        input_shape=(img_height, img_width, 1),
        num_blocks=hype_space['num_blocks'],
        num_layers_per_block=hype_space['num_layers_per_block'],
        growth_rate=hype_space['growth_rate'],
        dropout_rate=hype_space['dropout_rate'],
        compress_factor=hype_space['compress_factor'],
        num_filters=hype_space['num_filters'],
        num_classes=num_classes_exp,
        se_config=hype_space['se_config']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=7, monitor='val_loss'),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(RESULTS_DIR, 'best_model.keras'),
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3)
    ]
    
    # Treinamento
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        class_weight=class_weight,
        callbacks=callbacks
    )
    
    # Avaliação
    test_preds = model.predict(test_gen)
    y_pred = np.argmax(test_preds, axis=1)
    y_true = test_gen.classes
    
    # Métricas
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'model': model,
        'history': history.history,
        'accuracy': acc,
        'report': report,
        'confusion_matrix': cm.tolist()
    }