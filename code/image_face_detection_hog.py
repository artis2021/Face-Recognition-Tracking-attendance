# -*- coding: utf-8 -*-
"""

@author: arti
"""
#importing the required libraries
import cv2
import dlib


#loading the image to detect
image_to_detect = cv2.imread('trump-modi.jpg')

#create a grayscale image to pass into the dlib HOG detector
image_to_detect_gray = cv2.cvtColor(image_to_detect, cv2.COLOR_BGR2GRAY)

#load the pretrained HOG SVN model
face_detection_classifier = dlib.get_frontal_face_detector()

#detect all face locations using the HOG SVN classifier
all_face_locations = face_detection_classifier(image_to_detect,1)

#print the number of faces detected
print('There are {} no of faces in this image'.format(len(all_face_locations)))

#looping through the face locations
for index,current_face_location in enumerate(all_face_locations):
    #start and end co-ordinates
    left_x, left_y, right_x, right_y = current_face_location.left(),current_face_location.top(),current_face_location.right(),current_face_location.bottom()
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



