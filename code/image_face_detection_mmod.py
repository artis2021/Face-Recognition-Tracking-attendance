# -*- coding: utf-8 -*-
"""

@author: arti
"""
#importing the required libraries
import cv2
import dlib


#loading the image to detect
image_to_detect = cv2.imread('trump-modi.jpg')


# load the pretrained mmod model
# Download Model from https://github.com/davisking/dlib-models
# Dataset Author, Davis King, consisting of images from various datasets (contains 7220 images)

#load the pretrained MMOD model
face_detection_classifier = dlib.cnn_face_detection_model_v1('models/mmod_human_face_detector.dat')

#detect all face locations using the MMOD classifier
all_face_locations = face_detection_classifier(image_to_detect,1)

#print the number of faces detected
print('There are {} no of faces in this image'.format(len(all_face_locations)))

#looping through the face locations
for index,current_face_location in enumerate(all_face_locations):
    #start and end co-ordinates
    left_x, left_y, right_x, right_y = current_face_location.rect.left(),current_face_location.rect.top(),current_face_location.rect.right(),current_face_location.rect.bottom()
    #printing the location of current face
    print('Found face {} at left_x:{},left_y:{},right_x:{},right_y:{}'.format(index+1,left_x,left_y,right_x,right_y))
    #slicing the current face from main image
    current_face_image = image_to_detect[left_y:right_y,left_x:right_x]
    #showing the current face with dynamic title
    cv2.imshow("Face no "+str(index+1),current_face_image)
    #draw bounding box around the faces
    cv2.rectangle(image_to_detect,(left_x,left_y),(right_x,right_y),(0,255,0),2)

#show the image
cv2.imshow("faces in image",image_to_detect)
#keep the window waiting until we press a key
cv2.waitKey(0)
#close all windows
cv2.destroyAllWindows()
