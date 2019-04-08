import cv2 as cv
import signal
import os

class Face_Saver:

    UNKNOWN_FACE_DIR = 'uncropped_people_images'
    KNOWN_FACE_DIR = 'cropped_face_images'

    def __init__(self):
        signal.signal(signal.SIGINT, self.keyboardInterruptHandler)
        with open('id.txt') as f:
            self.photo_id = int(f.readline())
            print('Photo id', self.photo_id, 'is loaded.')
        self.face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


    def keyboardInterruptHandler(self, signal, frame):
    
        with open('id.txt', 'w') as f:
            f.write(str(self.photo_id))
            print('Photo id', self.photo_id, 'is saved.')
        print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(signal))
        exit(0)


    def save_face_from_directory(self, dir_path):
        
        dir_list = os.listdir(dir_path)

        if ('.DS_Store' in dir_list):
            dir_list.remove('.DS_Store')
            print('.DS_Store is removed.')

        for image_path in dir_list:
            image_path = dir_path + '/' + image_path
            self.save_face_image(image_path)
            os.remove(image_path)

        with open('id.txt', 'w') as f:
            print(self.photo_id)
            f.write(str(self.photo_id))
            print('Photo id', self.photo_id, 'is saved.')

        print('Facial data extraction is finished.')


    def save_face_image(self, dir_path):

        img = cv.imread(dir_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cropped_image = img[y:y+h, x:x+w]
            cv.imwrite(self.KNOWN_FACE_DIR + '/' + str(self.photo_id) + '.jpg', cropped_image)
            self.photo_id = self.photo_id + 1

        print('Completed a photo')

    def save_photo_id(self):
        with open('id.txt', 'w') as f:
            print(self.photo_id)
            f.write(str(self.photo_id))
            print('Photo id', self.photo_id, 'is saved.')
