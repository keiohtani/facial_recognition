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

def draw_boxes(pil_image, unknown_face_locations, unknown_face_encodings, known_face_encodings, known_face_names):
    draw = ImageDraw.Draw(pil_image)
    for (top, right, bottom, left), unknown_face_encoding in zip(unknown_face_locations, unknown_face_encodings):

        matches = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)

        name = "unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = str(known_face_names[first_match_index])
        
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom + text_height + 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom + text_height - 3), name, fill=(255, 255, 255, 255))
    del draw
    pil_image.show()
    return pil_image

def load_known_faces(known_face_dir):
    known_face_encodings = []
    known_face_names = []
    with open(ID_FILE, mode='r') as f:
        last_id = int(f.read())

    dir_list = os.listdir(known_face_dir)
    for i in dir_list:
        image_path = 'known_faces/' + str(i) + '/test.jpg'
        known_image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(known_image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(i)
    print('face data is loaded')
    return known_face_encodings, known_face_names
    
    
def compare_faces(known_face_encodings, known_face_names):
    unknown_image = face_recognition.load_image_file('sample.jpg')

    unknown_face_locations = face_recognition.face_locations(unknown_image)
    unknown_face_encodings = face_recognition.face_encodings(unknown_image)

    results = face_recognition.compare_faces(known_face_encodings, unknown_face_encodings[0])
    for (top, right, bottom, left), unknown_face_encoding in zip(unknown_face_locations, unknown_face_encodings):

        matches = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)

        name = "unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            print(name)
    
def save_face_image(face_locations):
    known_face_encodings, known_face_names = load_known_faces()
    # read id file to keep track of the last id for students
    with open(ID_FILE, mode='r') as f:
        person_id = int(f.readline())
    
    for (top, right, bottom, left) in face_locations:
        cropped_image = pil_image.crop((left - CROP_MARGIN, top - CROP_MARGIN, right + CROP_MARGIN, bottom + CROP_MARGIN))
        cropped_image.save('known_faces/' + str(person_id) + '.jpg')
        person_id = person_id + 1

    with open(ID_FILE, mode='w') as f:
        f.write(str(person_id))

def save_unknown_faces(unknown_face_locations, unknown_face_encodings):
    
    
    known_face_encodings, known_face_names = load_known_faces()

    # read id file to keep track of the last id for students
    with open(ID_FILE, mode='r') as f:
        person_id = int(f.readline())

    for (top, right, bottom, left), unknown_face_encoding in zip(unknown_face_locations, unknown_face_encodings):

        matches = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)
        print(matches)
        print("face found")
        if any(matches):
            None
        else:
            cropped_image = pil_image.crop((left - CROP_MARGIN, top - CROP_MARGIN, right + CROP_MARGIN, bottom + CROP_MARGIN))
            new_image_path = 'known_faces/' + str(person_id) + '.jpg'
            cropped_image.save(new_image_path)
            print('face data is saved to', str(new_image_path))
            person_id = person_id + 1

    print("all the face data in the photo is saved. ")

    with open(ID_FILE, mode='w') as f:
        f.write(str(person_id))

def save_unknown_faces_into_direcotry(unknown_face_dir, known_face_dir):
    dir_list = os.listdir(unknown_face_dir)

    # read id file to keep track of the last id for students
    with open(ID_FILE, mode='r') as f:
        person_id = int(f.readline())

    for image_path in dir_list:
        known_face_encodings, known_face_names = load_known_faces(known_face_dir)
        image = face_recognition.load_image_file(unknown_face_dir + '/' + image_path)
        unknown_face_locations = face_recognition.face_locations(image)
        unknown_face_encodings = face_recognition.face_encodings(image, unknown_face_locations)
        pil_image = Image.fromarray(image)

        if (not known_face_encodings):
            for (top, right, bottom, left), unknown_face_encoding in zip(unknown_face_locations, unknown_face_encodings):
                cropped_image = pil_image.crop((left - CROP_MARGIN, top - CROP_MARGIN, right + CROP_MARGIN, bottom + CROP_MARGIN))
                new_image_path = 'known_faces/' + str(person_id) + '/test.jpg'
                print('a new face is found')
                if (not os.path.isdir('known_faces/' + str(person_id))):
                    os.mkdir('known_faces/' + str(person_id))
                cropped_image.save(new_image_path)
                print('face data is saved to', str(new_image_path))
                person_id = person_id + 1
        else: 
            for (top, right, bottom, left), unknown_face_encoding in zip(unknown_face_locations, unknown_face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)
                print(matches)
                if any(matches):
                    None
                else:
                    cropped_image = pil_image.crop((left - CROP_MARGIN, top - CROP_MARGIN, right + CROP_MARGIN, bottom + CROP_MARGIN))
                    new_image_path = 'known_faces/' + str(person_id) + '/test.jpg'
                    print('a new face is found')
                    if (not os.path.isdir('known_faces/' + str(person_id))):
                        os.mkdir('known_faces/' + str(person_id))
                        print("make a new dir")
                    cropped_image.save(new_image_path)
                    print('face data is saved to', str(new_image_path))
                    person_id = person_id + 1

    print("all the face data in the photo is saved. ")
    with open(ID_FILE, mode='w') as f:
        f.write(str(person_id))

if __name__ == '__main__':
    # loading image
    save_unknown_faces_into_direcotry(UNKNOWN_FACE_DIR, KNOWN_FACE_DIR)

    # known_face_encodings, known_face_names = load_known_faces()
    # save_face_image(unknown_face_locations)
    # draw_boxes(pil_image, unknown_face_locations, unknown_face_encodings, known_face_encodings, known_face_names)
    # compare_faces(known_face_encodings, known_face_names)
    # save_unknown_faces(unknown_face_locations, unknown_face_encodings)
