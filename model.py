# https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/


from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(INPUT_SIZE, INPUT_SIZE))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

INPUT_SIZE = 50

model = ResNet50(weights='imagenet', include_top=False)

# from keras.layers import Dense, Flatten
# from keras.models import Sequential

# model = Sequential()
# model.add(conv_base)
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

img1_representation = l2_normalize(model.predict(preprocess_image('img1.jpg'))[0,:])
img2_representation = l2_normalize(model.predict(preprocess_image('img5.jpg'))[0,:])

euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)

print(euclidean_distance)

threshold = 1
if euclidean_distance < threshold:
    print("verified... they are same person")
else:
    print("unverified! they are not same person!")

