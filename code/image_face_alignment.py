# -*- coding: utf-8 -*-
"""

@author: arti
"""
#importing the required libraries
import cv2
import dlib


#loading the image to detect
image_to_detect = cv2.imread('elon.jpg')

#load the pretrained HOG SVN model
face_detection_classifier = dlib.get_frontal_face_detector()

# detect all face locations using the HOG SVN classifier
# temporarily upscale the image to 1 time(s), so that the face detection is easy
all_face_locations = face_detection_classifier(image_to_detect,1)

#print the number of faces detected
print('There are {} no of faces in this image'.format(len(all_face_locations)))

# object to hold the 5 face landmark points for every face
# each object will contain the box points and list of facial points.
face_landmarks = dlib.full_object_detections()

# download the trained facial shape predictor from
# http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
# load shape predictor to predict face landmark points of individual facial structures
face_shape_predictor = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')

#looping through the face locations
for index,current_face_location in enumerate(all_face_locations):
    #looping through all face detections and append shape predictions
    face_landmarks.append(face_shape_predictor(image_to_detect, current_face_location))

# get all face chips (rectangular cut image) using dlib.get_face_chips()
# inputs are source image and a full_object_detections object that reference faces
# returns the faces list rotated upright and scaled to 150x150 pixels
# optional arguments eg: dlib.get_face_chips(image_to_detect, face_landmarks, size=160, padding=0.25)
all_face_chips = dlib.get_face_chips(image_to_detect,face_landmarks)    

#loop through the face chips and show them
for index,current_face_chip in enumerate(all_face_chips):
    #show the face chip
    cv2.imshow("Face no "+str(index+1),current_face_chip)

#keep the window waiting until we press a key
cv2.waitKey(0)
#close all windows
cv2.destroyAllWindows()

