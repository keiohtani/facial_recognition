# https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
# https://medium.com/@sumantrajoshi/face-recognizer-application-using-a-deep-learning-model-python-and-keras-2873e9aa6ab3

from preprocess import *
from model import *
import cv2
import os
import numpy as np


class Facial_Recogition():

    def __init__(self):
        self.model, self.input_size = Inception_Model()
        # self.model, self.input_size = VGG_face_model()
        self.dir_path = 'face_database'
        self.image_dir_list = os.listdir(self.dir_path)

        if ('.DS_Store' in self.image_dir_list):
            self.image_dir_list.remove('.DS_Store')
            print('.DS_Store is removed.')

        self.np_dataset = self.load_database()
        self.face_haarcascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')
        print('Database is loaded.')

    def calculate_distance(self, img1, img2):

        epsilon = 0.40

        img1_representation = self.model.predict(
            preprocess_image_from_path(img1, self.input_size))[0, :]
        img2_representation = self.model.predict(
            preprocess_image_from_path(img2, self.input_size))[0, :]

        # euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
        cosine_similarity = self.findCosineDistance(
            img1_representation, img2_representation)
        return cosine_similarity

    def load_database(self):

        dataset = []

        for stored_image in self.image_dir_list:
            img_representation = self.model.predict(
                preprocess_image_from_path('face_database/' + stored_image, self.input_size))[0, :]
            dataset.append(img_representation)

        return np.array(dataset)

    def recognize_vector_euclidean_distance(self, test_representation):

        epsilon = 120
        euclidean_distance = self.np_dataset - test_representation
        euclidean_distance = np.sum(
            euclidean_distance * euclidean_distance, axis=1)
        euclidean_distance = np.sqrt(euclidean_distance)

        if euclidean_distance.min() < epsilon:
            index = np.argmin(euclidean_distance)
            name = self.image_dir_list[index]
            return name

    def recognize_vector_cosine_distance(self, test_representation):

        epsilon = 0.4
        a = np.matmul(self.np_dataset, test_representation)
        b = np.sum(self.np_dataset * self.np_dataset, axis=1)
        c = np.sum(np.multiply(test_representation, test_representation))
        x = 1 - (a / (np.sqrt(b) * np.sqrt(c)))

        if x.min() < epsilon:
            index = np.argmin(x)
            name = self.image_dir_list[index]
            return name

    def capture(self):

        cap = cv2.VideoCapture(0)
        end_flag, c_frame = cap.read()
        cv2.imwrite('captured.jpg', c_frame)
        return c_frame

    def haarcascade_crop_face(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_haarcascade.detectMultiScale(gray, 1.3, 5)
        cropped_image = []

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cropped_image = img[y:y+h, x:x+w]

        return cropped_image, img, faces

    def recognize(self, image_path):
        test_representation = self.model.predict(
            preprocess_image_from_path(image_path, self.input_size))[0, :]
        self.recognize_vector_cosine_distance(test_representation)

    def recognize_capture(self):
        c_frame = self.capture()
        img, c_frame, faces = self.haarcascade_crop_face(c_frame)
        if img != []:
            test_representation = self.model.predict(
                preprocess_opencv_image(img, self.input_size))[0, :]
            self.recognize_vector_euclidean_distance(test_representation)

    def recognize_realtime(self):

        INTERVAL = 100
        DEVICE_ID = 0
        ESC_KEY = 27
        WINDOW_NAME = 'facial recognition'
        SCALE = 1.0
        COLOR = (0, 0, 255)

        cap = cv2.VideoCapture(DEVICE_ID)
        end_flag, frame = cap.read()
        print('Press esc to exit.')

        while end_flag:

            key = cv2.waitKey(INTERVAL)
            if key == ESC_KEY:
                break

            end_flag, c_frame = cap.read()
            cropped_image, c_frame, faces = self.haarcascade_crop_face(c_frame)

            if cropped_image != []:
                test_representation = self.model.predict(
                    preprocess_opencv_image(cropped_image, self.input_size))[0, :]
                name = self.recognize_vector_cosine_distance(
                    test_representation)
                cv2.putText(
                    c_frame, name, (faces[0, 0], faces[0, 1]), cv2.FONT_HERSHEY_DUPLEX, SCALE, COLOR)

            cv2.imshow(WINDOW_NAME, c_frame)

        cv2.imwrite('last_frame.jpg', c_frame)
        cv2.destroyAllWindows()
        cap.release()


if __name__ == '__main__':

    test_image_path = 'test.jpg'
    fr = Facial_Recogition()
    # fr.recognize(test_image_path)
    fr.recognize_realtime()
    # fr.recognize_capture()
