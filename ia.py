import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from keras.models import load_model

def predict_image(model, image_path, img_height, img_width):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255
    prediction = model.predict(img_array)
    return prediction

def get_class_label(prediction, class_indices):
    class_label = None
    max_prob = np.max(prediction)
    for label, index in class_indices.items():
        if prediction[0][index] == max_prob:
            class_label = label
            break
    return class_label

model = load_model('modelo.h5')

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    './data',
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical'
)

class_indices = train_generator.class_indices

image_path = '1.jpg'

prediction = predict_image(model, image_path, 150, 150)
class_label = get_class_label(prediction, class_indices)

print(f"A imagem ({image_path}) tem {np.max(prediction) * 100:.2f}% de probabilidade de ser um {class_label}")