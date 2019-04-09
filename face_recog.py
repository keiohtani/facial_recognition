import signal
from PIL import Image
import cv2
from align import AlignDlib


class Face_Saver:

    KNOWN_FACE_DIR = 'cropped_face_images'

    def __init__(self):
        signal.signal(signal.SIGINT, self.keyboardInterruptHandler)
        self.alignment = AlignDlib('models/landmarks.dat')
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
        
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        # Detect face and return bounding box (type: dlib.rectangle)
        bounding_boxes = self.alignment.getAllFaceBoundingBoxes(image)
        
        for bounding_box in bounding_boxes:
            # Transform image using specified face landmark indices and crop image to 224x224
            cropped_image = self.alignment.align(224, image, bounding_box, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
            cv2.imwrite('cropped_face_images/' + str(self.photo_id) + '.jpg', cropped_image)
            self.photo_id = self.photo_id + 1

        print('Completed a photo')
                

    def save_photo_id(self):
        with open('id.txt', 'w') as f:
            print(self.photo_id)
            f.write(str(self.photo_id))
            print('Photo id', self.photo_id, 'is saved.')
