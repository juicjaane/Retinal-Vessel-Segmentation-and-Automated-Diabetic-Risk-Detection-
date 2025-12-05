import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from preprocessing import preprocess_input_image

def build_model(num_classes):
    """
    Builds the VGG16-based transfer learning model.
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train_model(data_dir, batch_size=32, epochs=20):
    """
    Trains the classifier.
    """
    # Data Augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_image, # Use our custom preprocessing
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_image,
        validation_split=0.2
    )

    print("Loading Training Data...")
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    print("Loading Validation Data...")
    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    num_classes = len(train_generator.class_indices)
    model = build_model(num_classes)
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("Starting Training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs,
        callbacks=[early_stopping]
    )
    
    return model, history, validation_generator

def evaluate_model(model, validation_generator):
    """
    Evaluates the model and plots confusion matrix.
    """
    Y_pred = model.predict(validation_generator, validation_generator.samples // validation_generator.batch_size + 1)
    y_pred = np.argmax(Y_pred, axis=1)
    
    print('Confusion Matrix')
    cm = confusion_matrix(validation_generator.classes, y_pred)
    print(cm)
    
    target_names = list(validation_generator.class_indices.keys())
    print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    # Example usage
    DATA_DIR = '../datasets/' # Adjust path relative to this script
    if os.path.exists(DATA_DIR):
        model, history, val_gen = train_model(DATA_DIR)
        evaluate_model(model, val_gen)
        model.save('diabetic_retinopathy_model.h5')
    else:
        print(f"Data directory not found at {DATA_DIR}. Please check the path.")
