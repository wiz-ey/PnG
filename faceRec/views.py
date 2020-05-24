from django.shortcuts import render

import tensorflow as tf
import cv2
from cv2 import imread
from cv2 import CascadeClassifier

import numpy as np
from tensorflow.keras.models import load_model

from OutOfStock.forms import FaceNetForm

# Create your views here.

classifier = CascadeClassifier('faceRec/haarcascade_frontalface_default.xml')
model = load_model('faceRec/facenet3.7.h5', custom_objects={"tf":tf})
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

create_database('sarthak','faceRec/test2.jpeg')

def identify(img):
    embedding=conversion_to_embedding(conversion(img,classifier))
    name,permission=recognition(embedding)
    if permission==True:
        return name
    else:
        name = 'imposter'
        return name

def rec_face(request):
    if request.method == "POST":
        form = FaceNetForm(request.POST)

        if form.is_valid():
            name = identify(read(form.cleaned_data['image']))
            return render(request, 'OutOfStock/face_identify.html', {'name':name})
    else:
        form = FaceNetForm()
    return render(request, 'OutOfStock/face_identify.html', {'form':form})
