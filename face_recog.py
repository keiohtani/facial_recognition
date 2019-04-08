import face_recognition
import signal
from PIL import Image


class Face_Saver: 

    KNOWN_FACE_DIR = 'cropped_face_images'

    def __init__(self):
        signal.signal(signal.SIGINT, self.keyboardInterruptHandler)
        with open('id.txt') as f:
            self.photo_id = int(f.readline())
            print('Photo id', self.photo_id, 'is loaded.')
        

    def keyboardInterruptHandler(self, signal, frame):
        
        with open('id.txt', 'w') as f:
            f.write(str(self.photo_id))
            print('Photo id', self.photo_id, 'is saved.')

        print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(signal))
        exit(0)


    def save_face_image(self, path):

        image = face_recognition.load_image_file(path)
        pil_image = Image.fromarray(image)
        unknown_face_locations = face_recognition.face_locations(image)
        
        for (top, right, bottom, left) in unknown_face_locations:
            cropped_image = pil_image.crop((left, top, right, bottom))
            cropped_image.save(self.KNOWN_FACE_DIR + '/' + str(self.photo_id) + '.jpg')
            self.photo_id = self.photo_id + 1
        
        print('Completed a photo')


    def save_photo_id(self):
        with open('id.txt', 'w') as f:
            print(self.photo_id)
            f.write(str(self.photo_id))
            print('Photo id', self.photo_id, 'is saved.')
