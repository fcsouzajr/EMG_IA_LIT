import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- configuração ---
data_dir = './data'
img_width = 150
img_height = 150
batch_size = 32
numero_classes = 2
epochs = 50

# --- geradores de dados ---
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# --- Carregamento dos dados ---
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset="training",
)
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset="validation",
)

# --- definição da arquitetura do modelo ---
model = keras.Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(img_height, img_width, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(numero_classes, activation="softmax"),
    ]
)
# --- compilação do modelo ---
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- treinamento do modelo ---
history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# --- salvando o modelo treinado ---
model.save('modelo.h5')
print('Modelo treinado com sucesso!')
