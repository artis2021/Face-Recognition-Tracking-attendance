# -*- coding: utf-8 -*-
"""

@author: arti
"""
#importing the required libraries
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine, euclidean
import cv2

def detect_extract_face(image_to_detect):
    
    #create an instance of MTCNN detector
    mtcnn_detector = MTCNN()
    
    #detect all face locations using the mtcnn dectector
    all_face_locations = mtcnn_detector.detect_faces(image_to_detect)
    
    #print the number of faces detected
    print('There are {} no of faces in the image'.format(len(all_face_locations)))
    #print(all_face_locations)
    
    
    #looping through the face locations
    for index,current_face_location in enumerate(all_face_locations):
        #splitting the tuple to get the four position values of current face
        x,y,width,height = current_face_location['box']
        #start co-ordinates
        left_x, left_y = x,y
        #end co-ordinates
        right_x, right_y = x+width, y+height
        #printing the location of current face
        #print('Found face {} at left_x:{},left_y:{},right_x:{},right_y:{}'.format(index+1,left_x,left_y,right_x,right_y))
        #slicing the current face from main image
        current_face_image = image_to_detect[left_y:right_y,left_x:right_x]
        #convert image from plt array to pil array
        current_face_image = Image.fromarray(current_face_image)
        #resize image to prefered size
        current_face_image = current_face_image.resize((224,224))
        #converting to numpy array
        current_face_image_np_array = np.asarray(current_face_image)
        #return array
        return current_face_image_np_array
    



#get the video
video_stream = cv2.VideoCapture('modi.mp4')
#video_stream = cv2.VideoCapture(0)

#loop through every frame in the video
while True:
    #get the current frame from the video stream as an image
    ret,image_to_classify = video_stream.read()
    
        
    #collecting the array of faces into a single list
    sample_faces = [detect_extract_face(plt.imread('modi/1.jpg')),
                    detect_extract_face(image_to_classify)]     
    
    if sample_faces[1] is not None:
        
        #convert to float 32 array
        sample_faces = np.asarray(sample_faces,'float32')
        #preprocess the array 
        sample_faces = preprocess_input(sample_faces, version=2)
        #create the vgg face model
        vggface_model = VGGFace(include_top=False, model='resnet50', input_shape=(224,224,3), pooling='avg')   
                
        #face feature (embeddings) extraction
        sample_faces_embeddings = vggface_model.predict(sample_faces)
        
        #the face that need to be verified
        image_to_be_verified = sample_faces_embeddings[0]
        
        #the faces that will be verfied against
        image_to_verify = sample_faces_embeddings[1]
        
        #calculate the face distance
        face_distance = cosine(image_to_be_verified, image_to_verify)
        #face_distance = euclidean(image_to_be_verified, image_to_verify)
        
        
        cv2.putText(image_to_classify,str(face_distance),(50,50),cv2.FONT_HERSHEY_PLAIN,2.5,(0,255,0),2)
    
        #show the image in window
        cv2.imshow("Distance", cv2.resize(image_to_classify, (224,224)))
        cv2.waitKey(5)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
#release the stream and cam
#close all opencv windows open
video_stream.release()
cv2.destroyAllWindows()  







        
         
 