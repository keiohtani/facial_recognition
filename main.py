# https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
# https://github.com/iwantooxxoox/Keras-OpenFace
# https://medium.com/@sumantrajoshi/face-recognizer-application-using-a-deep-learning-model-python-and-keras-2873e9aa6ab3
from model import *
import cv2
import os
import numpy as np
from downloader.align import AlignDlib
from keras.preprocessing.image import load_img, img_to_array
from cv2 import resize
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class Facial_Recogition():

    def __init__(self, architecture='Inception', alg='svc', threshold=0.4, visualize=False, n_neighbors=1):
        '''
        initializing Facial_Recognition
        :param architecture: architecture of NN model, either Inception or VGG16
        :type architecture: str
        :param alg: algrithm for sklearn model, either svc or knn
        :type alg: str
        :param threshold: threshold for sklearn prediction
        :type threshold: float
        :param visualize: whether to visualize training sets or not
        :type visualize: bool
        '''
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.architecture = architecture
        self.alg = alg
        self.photo_id = 0
        if architecture == 'Inception':
            self.model = create_model()
            self.input_size = 96
        elif architecture == 'VGG16':
            self.model = VGG_face_model()
            self.input_size = 224
        else:
            print('A valid name for model needs to be passed.')
            exit()

        if alg == 'svc':
            self.skmodel = LinearSVC()
        elif alg == 'knn':
            self.skmodel = KNeighborsClassifier(n_neighbors=n_neighbors)
        else:
            print('A valid name for alg needs to be passed.')
            exit()

        dir_path = 'training_images'
        X_train, y_train, images = self.load_data(dir_path, visualize=visualize)
        self.skmodel.fit(X_train, y_train)
        self.alignment = AlignDlib('downloader/landmarks.dat')

    def recognize_real_time(self):
        '''
        Starting a video capture
        '''

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
            c_frame = self.draw_bounding_box(c_frame)
            cv2.imshow(WINDOW_NAME, c_frame)

        cv2.imwrite('last_frame.jpg', c_frame)
        cv2.destroyAllWindows()
        cap.release()

    def draw_bounding_box(self, c_frame):
        '''
        draw a bounding box and predict indivisual using scikit learn's model defined above
        :param c_frame: an image taken from OpenCV's video capture
        :type path: numpy.ndarray
        :return: np array of preprocessed image
        :rtype: numpy.ndarray
        '''

        RED = (0, 0, 255)
        BLUE = (255, 0, 0)
        SCALE = 1

        # Detect face and return bounding box (type: dlib.rectangle)
        bounding_box = self.alignment.getLargestFaceBoundingBox(c_frame)

        if bounding_box != None:
            # Transform image using specified face landmark indices and crop image to 96x96
            cropped_image = self.alignment.align(
                self.input_size, c_frame, bounding_box, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
            cv2.rectangle(c_frame, (bounding_box.left(), bounding_box.top(
                )), (bounding_box.right(), bounding_box.bottom()), BLUE, 2)
            cropped_image = cropped_image[..., ::-1]
            cropped_image = self.preprocess(cropped_image)
            name = self.predict(cropped_image)
            print(name)
            cv2.putText(
                c_frame, name, (bounding_box.left(), bounding_box.top()), cv2.FONT_HERSHEY_DUPLEX, SCALE, RED)

        return c_frame

    def predict(self, image):
        '''
        predict indivisual using scikit learn's model defined above
        :param image: 
        :type path: numpy.ndarray (width, height, 3)
        :return: name of individual if the prediction was above threshold
        :rtype: str
        '''

        name = 'no_name'
        test_representation = self.model.predict(image)[0, :]
        if self.alg == 'svc':
            predictions = self.skmodel.decision_function(
                [test_representation])
            if predictions.max() > self.threshold:
                name = self.skmodel.predict([test_representation])[0]
        elif self.alg == 'knn':
            # predictions = self.skmodel.predict_proba([test_representation])
            distances, indices = self.skmodel.kneighbors(X=[test_representation], n_neighbors=n_neighbors, return_distance=True)
            min_dis = distances.min()
            print(min_dis)
            if min_dis < self.threshold:
                name = self.skmodel.predict([test_representation])[0]
        return name

    def load_data(self, dir_path, visualize=False):
        '''
        Loading images from directory and embeding using NN model defined above
        :param dir_path: path to images (folder needs to have folders)
        :type dir_path: str
        :return X: np array containing np arraies of image
        :rtype X: numpy.ndarray Shape: (numitems, either 128 or 2622 depending on which model to use)
        :return y: np array containing labels
        :rtype y: numpy.ndarray Shape: (numitems, 1)
        '''

        name_folders_dir = os.listdir(dir_path)
        X = []
        y = []
        images = []

        if ('.DS_Store' in name_folders_dir):
            name_folders_dir.remove('.DS_Store')
            print('.DS_Store is removed.')

        for name_folder_dir in name_folders_dir:
            name_list = os.listdir(os.path.join(dir_path, name_folder_dir))
            for stored_image in name_list:
                ext = os.path.splitext(stored_image)[1]
                if ext == '.jpg' or ext == '.jpeg':
                    path = os.path.join(dir_path, name_folder_dir, stored_image)
                    image = load_img(path, target_size=(
                        self.input_size, self.input_size))
                    image_array = img_to_array(image)
                    preprocessed_image = self.preprocess(image_array)
                    img_representation = self.model.predict(preprocessed_image)[0, :]
                    X.append(img_representation)
                    y.append(name_folder_dir)
                    images.append(image)

        if visualize:
            self.visualize(X, y)

        return X, y, images

    def preprocess(self, image):
        '''
        preprocess image either passed as an argument or loaded from the path
        :param path: path to test images (folder needs to have folders) or np array of image
        :type path: str or numpy.ndarray Shape(width, height, 3)
        :return: np array of preprocessed image
        :rtype: numpy.ndarray Shape(input_size * input_size * 3, 1)
        '''
        image = (image / 255.).astype(np.float32)
        image = np.expand_dims(image, axis=0)
        return image

    def test(self, path):
        '''
        Load data from path and test the accuracy of the model
        :param path: path to test images (folder needs to have folders)
        :type path: str
        '''
        X, y, images = self.load_data(path, visualize=False)
        names = []
        for image in images:
            image_array = img_to_array(image)
            preprocessed_image = self.preprocess(image_array)
            names.append(self.predict(preprocessed_image))
        match = set(names) & set(y)
        accuracy = len(match) / len(y)
        print(accuracy)
        return accuracy

    # http://krasserm.github.io/2018/02/07/deep-face-recognition/

    def visualize(self, X, y):
        '''
        Visualize X and y 
        :param X: np.ndarray of 128 dimensional array
        :type X: numpy.ndarray
        :param y: np.ndarray of labels
        :type y: numpy.ndarray
        '''
        X_embedded = TSNE(n_components=2).fit_transform(X)
        targets = np.array(y)
        for i, t in enumerate(set(targets)):
            idx = targets == t
            plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

        plt.legend(bbox_to_anchor=(1, 1))
        plt.show()


def grid_search():
    thresh = 0.5
    n_neighbors = 1
    temp = 0.1
    li = []
    big_list = []
    for j in range(1, 10):
        for i in range(10):
            thresh = temp * i
            fr = Facial_Recogition(architecture='Inception',
                                alg='knn', threshold=thresh, visualize=False, n_neighbors=j)
            accuracy = fr.test('test_images')
            li.append(accuracy)
        big_list.append(li)
    print(big_list)

if __name__ == '__main__':

    # grid_search()

    thresh = 0.6
    n_neighbors = 1
    fr = Facial_Recogition(architecture='Inception',
            alg='knn', threshold=thresh, visualize=False, n_neighbors=n_neighbors)
    # accuracy = fr.test('test_images')
    fr.recognize_real_time()