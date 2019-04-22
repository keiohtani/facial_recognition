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
from sklearn.svm import SVC


class Facial_Recogition():

    def __init__(self, architecture='VGG16', alg='svc'):
        self.architecture = architecture
        self.alg = alg
        if architecture == 'Inception':
            self.model = Inception_Model()
            self.input_size = 96
        elif architecture == 'VGG16':
            self.model = VGG_face_model()
            self.input_size = 224
        else:
            print('A valid name for model needs to be passed.')
        if alg=='svc':
            self.svc = SVC()
            dir_path = 'svc_face_database'
            self.fit(dir_path)
        else: 
            dir_path = 'face_database'
            self.np_dataset = self.load_database(dir_path)

        # self.image_dir_list = os.listdir(dir_path)
        # if ('.DS_Store' in self.image_dir_list):
        #     self.image_dir_list.remove('.DS_Store')
        #     print('.DS_Store is removed.')
        
        print('Database is loaded.')
        self.alignment = AlignDlib('models/landmarks.dat')

    def load_database(self, dir_path):

        dataset = []
        image_dir_list = os.listdir(dir_path)
        for stored_image in image_dir_list:
            img_representation = self.model.predict(
                self.preprocess(os.path.path.join(dir_path, stored_image)))[0, :]
            dataset.append(img_representation)

        return np.array(dataset)

    def find_distance(self, test_representation):
        if self.alg == 'cosine':
            epsilon = 0.023 if self.architecture == 'Inception' else 0.3
            a = np.matmul(self.np_dataset, test_representation)
            b = np.sum(self.np_dataset * self.np_dataset, axis=1)
            c = np.sum(np.multiply(test_representation, test_representation))
            distance = 1 - (a / (np.sqrt(b) * np.sqrt(c)))

        elif self.alg == 'euclidean':
            epsilon = 2.5 if self.architecture == 'Inception' else 120
            euclidean_distance = self.np_dataset - test_representation
            euclidean_distance = np.sum(
                euclidean_distance * euclidean_distance, axis=1)
            distance = np.sqrt(euclidean_distance)

        else:
            print('Pass either cosine or euclidean for distance')

        if distance.min() < epsilon:
            index = np.argmin(distance)
            name = self.image_dir_list[index]
            return name

    def recognize_from_path(self, image_path):
        test_representation = self.model.predict(
            self.preprocess(image_path))[0, :]
        name = self.find_distance(test_representation)
        print(name)

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
            if self.alg == 'svc':
                c_frame = self.predict_svc(c_frame)
            else: 
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
            name = self.find_distance(test_representation)
            cv2.putText(
                c_frame, name, (bounding_box.left(), bounding_box.top()), cv2.FONT_HERSHEY_DUPLEX, SCALE, RED)

        return c_frame

    def predict_svc(self, c_frame):
    
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
            # name = self.find_distance(test_representation)
            name = self.svc.predict([test_representation])
            print(name)
            cv2.putText(
                c_frame, name[0], (bounding_box.left(), bounding_box.top()), cv2.FONT_HERSHEY_DUPLEX, SCALE, RED)

        return c_frame

    def fit(self, dir_path):

        name_folders_dir = os.listdir(dir_path)
        X = []
        y = []

        for name_folder_dir in name_folders_dir:
            name_list = os.listdir(os.path.join(dir_path, name_folder_dir))
            for stored_image in name_list:
                img_representation = self.model.predict(
                    self.preprocess(os.path.join(dir_path, name_folder_dir, stored_image)))[0, :]
                X.append(img_representation)
            y.append(name_folder_dir)
        self.svc.fit(X, y)

    def preprocess(self, image):
        if type(image) is str:
            img = load_img(image, target_size=(
                self.input_size, self.input_size))
            img = img_to_array(img)
        else:
            img = resize(image, (self.input_size, self.input_size))
        img = np.expand_dims(img, axis=0)
        # img = preprocess_input(img) if self.architecture == 'VGG16' else img
        return img


if __name__ == '__main__':

    test_image_path = 'test.jpg'
    fr = Facial_Recogition(architecture='VGG16', alg='svc')
    # fr.recognize_from_path(test_image_path)
    fr.recognize()

