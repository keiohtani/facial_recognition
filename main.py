# https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
# https://medium.com/@sumantrajoshi/face-recognizer-application-using-a-deep-learning-model-python-and-keras-2873e9aa6ab3
from model import *
import cv2
import os
import numpy as np
from align import AlignDlib
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from cv2 import resize


class Facial_Recogition():

    def __init__(self, architecture='VGG16'):
        self.architecture = architecture
        if architecture == 'Inception':
            self.model = Inception_Model()
            self.input_size = 96
        elif architecture == 'VGG16':
            self.model = VGG_face_model()
            self.input_size = 224
        else:
            print('A valid name for model needs to be passed.')
        self.dir_path = 'face_database'
        self.image_dir_list = os.listdir(self.dir_path)

        if ('.DS_Store' in self.image_dir_list):
            self.image_dir_list.remove('.DS_Store')
            print('.DS_Store is removed.')

        self.np_dataset = self.load_database()
        print('Database is loaded.')
        self.alignment = AlignDlib('models/landmarks.dat')

    def load_database(self):

        dataset = []
        for stored_image in self.image_dir_list:
            img_representation = self.model.predict(
                self.preprocess('face_database/' + stored_image))[0, :]
            dataset.append(img_representation)

        return np.array(dataset)

    def find_distance(self, test_representation, alg='cosine'):
        if alg == 'cosine':
            epsilon = 0.023 if self.architecture == 'Inception' else 0.3 
            a = np.matmul(self.np_dataset, test_representation)
            b = np.sum(self.np_dataset * self.np_dataset, axis=1)
            c = np.sum(np.multiply(test_representation, test_representation))
            distance = 1 - (a / (np.sqrt(b) * np.sqrt(c)))

        elif alg == 'euclidean':
            epsilon = 2.5 if self.architecture == 'Inception' else 100
            euclidean_distance = self.np_dataset - test_representation
            euclidean_distance = np.sum(
                euclidean_distance * euclidean_distance, axis=1)
            distance = np.sqrt(euclidean_distance)
        
        else:
            print('Pass either cosine or euclidean for alg')

        print(distance)
        print(self.image_dir_list)

        if distance.min() < epsilon:
            index = np.argmin(distance)
            name = self.image_dir_list[index]
            return name

    def recognize(self, image_path):
        test_representation = self.model.predict(
            self.preprocess(image_path))[0, :]
        self.find_distance(test_representation)

    def recognize(self):

        INTERVAL = 100
        DEVICE_ID = 0
        ESC_KEY = 27
        WINDOW_NAME = 'facial recognition'

        cap = cv2.VideoCapture(DEVICE_ID)
        end_flag, frame = cap.read()
        print('Press esc to exit.')

        while end_flag:

            key = cv2.waitKey(INTERVAL)
            if key == ESC_KEY:
                break

            end_flag, c_frame = cap.read()
            c_frame = self.predict(c_frame)
            cv2.imshow(WINDOW_NAME, c_frame)

        cv2.imwrite('last_frame.jpg', c_frame)
        cv2.destroyAllWindows()
        cap.release()
    
    def predict(self, c_frame):

        RED = (0, 0, 255)
        BLUE = (255, 0, 0)
        SCALE = 1

        # Detect face and return bounding box (type: dlib.rectangle)
        bounding_box = self.alignment.getLargestFaceBoundingBox(c_frame)

        if bounding_box != None:
            cv2.rectangle(c_frame, (bounding_box.left(), bounding_box.top(
            )), (bounding_box.right(), bounding_box.bottom()), BLUE, 2)
            # Transform image using specified face landmark indices and crop image to 96x96
            cropped_image = self.alignment.align(
                self.input_size, c_frame, bounding_box, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
            test_representation = self.model.predict(
                self.preprocess(cropped_image))[0, :]
            name = self.find_distance(
                test_representation)
            name = self.find_distance(
                test_representation, alg='euclidean')
            cv2.putText(
                c_frame, name, (bounding_box.left(), bounding_box.top()), cv2.FONT_HERSHEY_DUPLEX, SCALE, RED)
        
        return c_frame

    def preprocess(self, image):
        if type(image) is str:
            img = load_img(image, target_size=(
                self.input_size, self.input_size))
            img = img_to_array(img)
        else:
            img = resize(image, (self.input_size, self.input_size))
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img) if self.architecture == 'VGG16' else img
        return img


if __name__ == '__main__':

    test_image_path = 'test.jpg'
    fr = Facial_Recogition(architecture='Inception')
    # fr.recognize(test_image_path)
    fr.recognize()

