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
        self.realtime = True

        if ('.DS_Store' in self.image_dir_list):
            self.image_dir_list.remove('.DS_Store')
            print('.DS_Store is removed.')
        
        self.np_dataset = self.load_database()
        self.face_haarcascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        print('Database is loaded.')


    def calculate_distance(self, img1, img2):

        epsilon = 0.40

        img1_representation = self.model.predict(preprocess_image_from_path(img1))[0,:]
        img2_representation = self.model.predict(preprocess_image_from_path(img2))[0,:]
        
        # euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
        cosine_similarity = self.findCosineDistance(img1_representation, img2_representation)
        return cosine_similarity


    def load_database(self):
        
        dataset = []

        for stored_image in self.image_dir_list:
            img_representation = self.model.predict(preprocess_image_from_path('face_database/' + stored_image))[0,:]
            dataset.append(img_representation)
        
        return np.array(dataset)
        

    def recognize_vector_euclidean_distance(self, test_image):

        epsilon = 120
        if self.realtime: 
            test_representation = self.model.predict(preprocess_opencv_image(test_image))[0,:]
        else:
            test_representation = self.model.predict(preprocess_image_from_path(test_image))[0,:]
        euclidean_distance = self.np_dataset - test_representation
        euclidean_distance = np.sum(euclidean_distance * euclidean_distance, axis=1)
        euclidean_distance = np.sqrt(euclidean_distance)

        if euclidean_distance.min() < epsilon:
            index = np.argmin(euclidean_distance)
            print(self.image_dir_list[index])


    def recognize_vector_cosine_distance(self, test_image):

        epsilon = 0.4
        if self.realtime: 
            test_representation = self.model.predict(preprocess_opencv_image(test_image))[0,:]
        else:
            test_representation = self.model.predict(preprocess_image_from_path(test_image))[0,:]
        a = np.matmul(self.np_dataset, test_representation)
        b = np.sum(self.np_dataset * self.np_dataset, axis = 1)
        c = np.sum(np.multiply(test_representation, test_representation))
        x = 1 - (a / (np.sqrt(b) * np.sqrt(c)))

        if x.min() < epsilon:
            index = np.argmin(x)
            print(self.image_dir_list[index])
    

    def capture(self):

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
        cv2.imwrite('hello.jpg', c_frame)
        cv2.destroyAllWindows()
        cap.release()
        return self.haarcascade_crop_face(c_frame)
    

    def haarcascade_crop_face(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_haarcascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            cropped_image = img[y:y+h, x:x+w]

        return cropped_image


if __name__ == '__main__':
    # test_image = 'test.jpg'
    img_dir = 'hello.jpg'
    fr = Facial_Recogition()
    test_image = fr.haarcascade_crop_face(cv2.imread(img_dir))
    fr.recognize_vector_cosine_distance(test_image)
    # fr.recognize_vector_euclidean_distance(test_image)