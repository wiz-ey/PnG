from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
from cv2 import imread
from cv2 import CascadeClassifier


classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('facenet3.7.h5')
database={}


def conversion(image,classifier):
    bboxes = classifier.detectMultiScale(image)
    x, y, width, height = bboxes[0]
    x2, y2 = x + width, y + height
    face=image[x-40:x2+40,y-40:y2+40]
    face=cv2.resize(face, (96,96), interpolation = cv2.INTER_AREA)
    img = face[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    img = np.transpose(img, (1, 2, 0))
    x = np.array([img])
    return x

def read(address):
    image=imread(address)
    return image

def conversion_to_embedding(img):
    embedding = model.predict(img)
    return embedding


def distance(embedding,database,identity):
    dist = np.linalg.norm(embedding - database[identity])
    return dist

def check(embedding,database,identity):
    if distance(embedding,database,identity)<0.7:
        print("Welcome " + str(identity))
        allow = True
    else:
        print("Unable to identify")
        allow = False
    return allow

def recognition(embedding):
    min_dist = 50
    for name in database:
        dist = distance(embedding,database,name)
        if dist<min_dist:
            min_dist = dist
            identity = name
    permission=check(embedding,database,identity)
    return name,permission

def create_database(name,address):
    database[name]=conversion_to_embedding(conversion(read(address),classifier))

def identify(img):
    embedding=conversion_to_embedding(conversion(img,classifier))
    name,permission=recognition(embedding)
    if permission==True:
        return name
    else:
        return 'imposter'

create_database('sarthak','test2.jpeg')
#img is the image captured by the camera
name=identify(read('test2.jpg'))
