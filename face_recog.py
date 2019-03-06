import face_recognition
import signal

ID_FILE = 'ids.txt'
UNKNOWN_FACE_DIR = 'unknown_faces'
KNOWN_FACE_DIR = 'known_faces'
CROP_MARGIN = 15
    
def save_face_image(face_locations, pil_image, photo_id):
    for (top, right, bottom, left) in face_locations:
        cropped_image = pil_image.crop((left - CROP_MARGIN, top - CROP_MARGIN, right + CROP_MARGIN, bottom + CROP_MARGIN))
        cropped_image.save('known_faces/' + str(photo_id) + '.jpg')
        photo_id = photo_id + 1
    print('Completed a photo')
    return photo_id

def save_face_from_directory(dir_path):
    
    def keyboardInterruptHandler(signal, frame):
        with open('id.txt', 'w') as f:
            f.write(str(photo_id))
            print('Photo id', photo_id, 'is saved.')
        print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(signal))
        exit(0)

    signal.signal(signal.SIGINT, keyboardInterruptHandler)

    with open('id.txt') as f:
        photo_id = int(f.readline())
        print('Photo id', photo_id, 'is loaded.')
    dir_list = os.listdir(dir_path)

    if ('.DS_Store' in dir_list):
        dir_list.remove('.DS_Store')
        print('.DS_Store is removed.')

    for image_path in dir_list:
        image_path = UNKNOWN_FACE_DIR + '/' + image_path
        image = face_recognition.load_image_file(image_path)
        unknown_face_locations = face_recognition.face_locations(image)
        unknown_face_encodings = face_recognition.face_encodings(image, unknown_face_locations)
        pil_image = Image.fromarray(image)
        photo_id = save_face_image(unknown_face_locations, pil_image, photo_id)
        os.remove(image_path)

    with open('id.txt', 'w') as f:
        f.write(str(photo_id))
        print('Photo id', photo_id, 'is saved.')
    print('Facial data extraction is finished.')

def save_face_from_path(path, photo_id):
    image = face_recognition.load_image_file(path)
    unknown_face_locations = face_recognition.face_locations(image)
    unknown_face_encodings = face_recognition.face_encodings(image, unknown_face_locations)
    pil_image = Image.fromarray(image)
    photo_id = save_face_image(unknown_face_locations, pil_image, photo_id)
    return photo_id

if __name__ == '__main__':
    save_face_from_directory(UNKNOWN_FACE_DIR)