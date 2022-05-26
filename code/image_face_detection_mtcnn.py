# -*- coding: utf-8 -*-
"""

@author: abhilash
"""
#importing the required libraries
import cv2
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt


#loading the image to detect
image_to_detect = plt.imread('trump-modi.jpg')

#create an instance of MTCNN detector
mtcnn_detector = MTCNN()

#detect all face locations using the mtcnn dectector
all_face_locations = mtcnn_detector.detect_faces(image_to_detect)

#print the number of faces detected
print('There are {} no of faces in this image'.format(len(all_face_locations)))
print(all_face_locations)

image_to_detect = cv2.cvtColor(image_to_detect,cv2.COLOR_BGR2RGB)

#looping through the face locations
for index,current_face_location in enumerate(all_face_locations):
    #splitting the tuple to get the four position values of current face
    x,y,width,height = current_face_location['box']
    #start co-ordinates
    left_x, left_y = x,y
    #end co-ordinates
    right_x, right_y = x+width, y+height
    #printing the location of current face
    print('Found face {} at left_x:{},left_y:{},right_x:{},right_y:{}'.format(index+1,left_x,left_y,right_x,right_y))
    #slicing the current face from main image
    current_face_image = image_to_detect[left_y:right_y,left_x:right_x]
    #showing the current face with dynamic title
    cv2.imshow("Face no "+str(index+1),current_face_image)
    #draw bounding box around the faces
    cv2.rectangle(image_to_detect,(left_x,left_y),(right_x,right_y),(0,255,0),2)
    #draw circles for every face keypoints
    keypoints = current_face_location['keypoints']
    cv2.circle(image_to_detect,(keypoints['left_eye']),5,(0,255,0),1)
    cv2.circle(image_to_detect,(keypoints['right_eye']),5,(0,255,0),1)
    cv2.circle(image_to_detect,(keypoints['nose']),5,(0,255,0),1)
    cv2.circle(image_to_detect,(keypoints['mouth_left']),5,(0,255,0),1)
    cv2.circle(image_to_detect,(keypoints['mouth_right']),5,(0,255,0),1)
    
#show the image
cv2.imshow("faces in image",image_to_detect)
#keep the window waiting until we press a key
cv2.waitKey(0)
#close all windows
cv2.destroyAllWindows()
