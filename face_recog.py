from PIL import Image, ImageDraw
import face_recognition
ID_FILE = 'ids.txt'

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

def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    with open(ID_FILE, mode='r') as f:
        last_id = int(f.read())
    for i in range(1, last_id):
        image_path = 'known_faces/' + str(i) + '.jpg'
        known_image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(known_image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(i)
    print('face data is loaded from', ID_FILE)
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
    margin = 15

    known_face_encodings, known_face_names = load_known_faces()
    # read id file to keep track of the last id for students
    with open(ID_FILE, mode='r') as f:
        person_id = int(f.readline())
    
    for (top, right, bottom, left) in face_locations:
        cropped_image = pil_image.crop((left - margin, top - margin, right + margin, bottom + margin))
        cropped_image.save('known_faces/' + str(person_id) + '.jpg')
        person_id = person_id + 1

    with open(ID_FILE, mode='w') as f:
        f.write(str(person_id))

def save_unknown_faces(unknown_face_locations, unknown_face_encodings):
    margin = 15
    
    known_face_encodings, known_face_names = load_known_faces()

    # read id file to keep track of the last id for students
    with open(ID_FILE, mode='r') as f:
        person_id = int(f.readline())

    for (top, right, bottom, left), unknown_face_encoding in zip(unknown_face_locations, unknown_face_encodings):

        matches = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)
        print("face found")
        if not all(matches):
            cropped_image = pil_image.crop((left - margin, top - margin, right + margin, bottom + margin))
            new_image_path = 'known_faces/' + str(person_id) + '.jpg'
            cropped_image.save(new_image_path)
            print('face data is saved to', str(new_image_path))
            person_id = person_id + 1

    print("all the face data in the photo is saved. ")

    with open(ID_FILE, mode='w') as f:
        f.write(str(person_id))

if __name__ == '__main__':

    # loading image
    image_path = 'sample_4.jpg'
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    pil_image = Image.fromarray(image)
    pil_image.show()

    # known_face_encodings, known_face_names = load_known_faces()
    # save_face_image(face_locations)
    # draw_boxes(pil_image, face_locations, face_encodings, known_face_encodings, known_face_names)
    # compare_faces(known_face_encodings, known_face_names)
    save_unknown_faces(face_locations, face_encodings)