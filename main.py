# https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/

from preprocess import *
from model import *
import cv2
import os
import numpy as np

class Facial_Recogition():
    
    def __init__(self):
        self.model = VGG_face_model()
        # remove the last two layers to get 128 dimensions vector 
        self.model = Model(inputs=self.model.layers[0].input, outputs=self.model.layers[-2].output)
        self.dir_path = 'face_database'
        self.image_dir_list = os.listdir(self.dir_path)
        if ('.DS_Store' in self.image_dir_list):
            self.image_dir_list.remove('.DS_Store')
            print('.DS_Store is removed.')
        self.np_dataset = self.load_database()
        print('Database is loaded.')


    def calculate_distance(self, img1, img2):

        epsilon = 0.40

        img1_representation = self.model.predict(preprocess_image(img1))[0,:]
        img2_representation = self.model.predict(preprocess_image(img2))[0,:]
        
        # euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
        cosine_similarity = self.findCosineDistance(img1_representation, img2_representation)
        return cosine_similarity


    def realtime_verification(self):
        INTERVAL = 30
        DEVICE_ID = 0  
        ESC_KEY = 27
        FRAME_RATE = 24
        WINDOW_NAME = 'facial recognition'

        cap = cv2.VideoCapture(DEVICE_ID)
        end_flag, c_frame = cap.read()
        # print('Press esc to exit.')

        # while end_flag == True:

        #     cv2.imshow(WINDOW_NAME, c_frame)

        #     key = cv2.waitKey(INTERVAL)
        #     if key == ESC_KEY:
        #         break

        #     end_flag, c_frame = cap.read()
        
        #exitting
        cv2.destroyAllWindows()
        cap.release()


    def load_database(self):
        
        dataset = []

        for stored_image in self.image_dir_list:
            img_representation = self.model.predict(preprocess_image('face_database/' + stored_image))[0,:]
            dataset.append(img_representation)
        
        return np.array(dataset)
        

    def recognize_vector_euclidean_distance(self, test_image):

        epsilon = 120
        test_representation = self.model.predict(preprocess_image(test_image))[0,:]

        euclidean_distance = self.np_dataset - test_representation
        euclidean_distance = np.sum(euclidean_distance * euclidean_distance, axis=1)
        euclidean_distance = np.sqrt(euclidean_distance)

        if euclidean_distance.min() < epsilon:
            index = np.argmin(euclidean_distance)
            print(self.image_dir_list[index])


    def recognize_vector_cosine_distance(self, test_image):

        epsilon = 0.4
        test_representation = self.model.predict(preprocess_image(test_image))[0,:]
        a = np.matmul(self.np_dataset, test_representation)
        b = np.sum(self.np_dataset * self.np_dataset, axis = 1)
        c = np.sum(np.multiply(test_representation, test_representation))
        x = 1 - (a / (np.sqrt(b) * np.sqrt(c)))

        if x.min() < epsilon:
            index = np.argmin(x)
            print(self.image_dir_list[index])


if __name__ == '__main__':
    test_image = 'test.jpg'
    fr = Facial_Recogition()
    fr.recognize_vector_cosine_distance(test_image)
    # fr.recognize_vector_euclidean_distance(test_image)