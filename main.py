# https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/

from preprocess import *
from model import *
import cv2
import os

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def verifyFace(model, img1, img2):
    epsilon = 0.40

    img1_representation = model.predict(preprocess_image(img1))[0,:]
    img2_representation = model.predict(preprocess_image(img2))[0,:]
    
    # euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    cosine_similarity = findCosineDistance(img1_representation, img2_representation)
    print(cosine_similarity)
    if(cosine_similarity < epsilon):
        print("verified... they are same person")
    else:
        print("unverified! they are not same person!")

def calculate_distance(model, img1, img2):
    epsilon = 0.40

    img1_representation = model.predict(preprocess_image(img1))[0,:]
    img2_representation = model.predict(preprocess_image(img2))[0,:]
    
    # euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    cosine_similarity = findCosineDistance(img1_representation, img2_representation)
    return cosine_similarity

def realtime_verification():
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



if __name__ == '__main__':
    test_image = 'test.jpg'
    model = VGG_face_model()
    face_model = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    dir_path = 'face_database'
    image_dir_list = os.listdir(dir_path)

    if ('.DS_Store' in image_dir_list):
        image_dir_list.remove('.DS_Store')
        print('.DS_Store is removed.')

    min_distance = 1.0

    for stored_image in image_dir_list:
        distance = calculate_distance(face_model, test_image, dir_path + '/' + stored_image)
        print(distance)
        if min_distance > distance:
            min_distance = distance
            name = stored_image
    
    if (min_distance < 0.4):
        print(name)
