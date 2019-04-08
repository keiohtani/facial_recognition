from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from cv2 import resize
import numpy as np


def preprocess_image_from_path(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def preprocess_opencv_image(image):
    img = resize(image, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
