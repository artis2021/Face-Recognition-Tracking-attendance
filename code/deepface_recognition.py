# -*- coding: utf-8 -*-
"""
Created on Tue May 24 18:53:07 2022

@author: arti
"""

from deepface import DeepFace


#detector backend="opencv","ssd","dlib" ,"mtcnn","retinaface"
#model_name="VGG-Face","Facenet","Facenet512","OpenFace","DeepFace"
#distance_metric="cosine","euclidean","euclidean_l2"

#face verification

face_recognition=DeepFace.find(img_path='testing/modi1.jpg',db_path='testing',detector_backend="opencv",model_name="VGG-Face",distance_metric="cosine")


print(face_recognition)

