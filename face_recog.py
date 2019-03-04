from PIL import Image, ImageDraw
import face_recognition
import os
import cv2
image_path = os.getcwd() + '/sample.jpg'
image = face_recognition.load_image_file(image_path)
face_locations = face_recognition.face_locations(image)

pil_image = Image.fromarray(image)

draw = ImageDraw.Draw(pil_image)

for (top, right, bottom, left) in face_locations:
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

del draw

pil_image.show()