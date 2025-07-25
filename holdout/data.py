from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def create_train_datagen():
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=False,
        fill_mode="constant"
    )

def create_val_test_datagen():
    return ImageDataGenerator(rescale=1./255)

def create_generators(train_dir, val_dir, test_dir, img_h, img_w, batch_sz, val_batch_sz):
    train_datagen = create_train_datagen()
    val_test_datagen = create_val_test_datagen()
    
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_h, img_w),
        color_mode='grayscale',
        batch_size=batch_sz,
        class_mode='categorical',
        shuffle=True
    )
    
    val_gen = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(img_h, img_w),
        color_mode='grayscale',
        batch_size=val_batch_sz,
        class_mode='categorical',
        shuffle=True
    )
    
    test_gen = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_h, img_w),
        color_mode='grayscale',
        batch_size=val_batch_sz,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_gen, val_gen, test_gen

def get_class_weights(train_gen):
    classes = np.unique(train_gen.classes)
    weights = compute_class_weight('balanced', classes=classes, y=train_gen.classes)
    return dict(enumerate(weights))