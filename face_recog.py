from PIL import Image, ImageDraw
import face_recognition
import os

#TODO
# - crop pictures 
# - add them into google photos

ID_FILE = 'ids.txt'
UNKNOWN_FACE_DIR = 'unknown_faces'
KNOWN_FACE_DIR = 'known_faces'
CROP_MARGIN = 15
    
def save_face_image(face_locations, pil_image, person_id):
    for (top, right, bottom, left) in face_locations:
        cropped_image = pil_image.crop((left - CROP_MARGIN, top - CROP_MARGIN, right + CROP_MARGIN, bottom + CROP_MARGIN))
        cropped_image.save('known_faces/' + str(person_id) + '.jpg')
        person_id = person_id + 1

def save_face_from_directory(dir_path):
    person_id = 1
    dir_list = os.listdir(dir_path)
    for image_path in dir_list:
        image = face_recognition.load_image_file(UNKNOWN_FACE_DIR + '/' + image_path)
        unknown_face_locations = face_recognition.face_locations(image)
        unknown_face_encodings = face_recognition.face_encodings(image, unknown_face_locations)
        pil_image = Image.fromarray(image)
        save_face_image(unknown_face_locations, pil_image, person_id)
        os.remove(image_path)

def save_face_from_path(path, photo_id):
    image = face_recognition.load_image_file(path)
    unknown_face_locations = face_recognition.face_locations(image)
    unknown_face_encodings = face_recognition.face_encodings(image, unknown_face_locations)
    pil_image = Image.fromarray(image)
    save_face_image(unknown_face_locations, pil_image, photo_id)

if __name__ == '__main__':
    save_face_from_directory(UNKNOWN_FACE_DIR)