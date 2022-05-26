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

def detect_extract_face(image_path_to_detect):
    
    #loading the image to detect
    image_to_detect = plt.imread(image_path_to_detect)
    
    #create an instance of MTCNN detector
    mtcnn_detector = MTCNN()
    
    #detect all face locations using the mtcnn dectector
    all_face_locations = mtcnn_detector.detect_faces(image_to_detect)
    
    #print the number of faces detected
    print('There are {} no of faces in the image {}'.format(len(all_face_locations),image_path_to_detect))
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
    
    
        
#collecting the array of faces into a single list
sample_faces = [detect_extract_face('modi/1.jpg'),
                detect_extract_face('modi/2.jpg'),
                detect_extract_face('modi/3.jpg'),
                detect_extract_face('biden/1.jpg')]        
        
#convert to float 32 array
sample_faces = np.asarray(sample_faces,'float32')
#preprocess the array 
sample_faces = preprocess_input(sample_faces, version=2)
#create the vgg face model
vggface_model = VGGFace(include_top=False, model='resnet50', input_shape=(224,224,3), pooling='avg')

print("input shape of the model")
print(vggface_model.inputs)      
        
#face feature (embeddings) extraction
sample_faces_embeddings = vggface_model.predict(sample_faces)

#the face that need to be verified
modi_face_1 = sample_faces_embeddings[0]

#the faces that will be verfied against
modi_face_2 = sample_faces_embeddings[1]
modi_face_3 = sample_faces_embeddings[2]
biden_face_1 = sample_faces_embeddings[3]

# Verify against the known photographs using cosine distance
print('********* cosine consider 0.5 as the threshold. if less then its a match*****')
print(cosine(modi_face_1, modi_face_2))
print(cosine(modi_face_1, modi_face_3))
print(cosine(modi_face_1, biden_face_1))

# Verify against the known photographs using euclidean distance
print('********* euclidean consider 100 as the threshold. if less then its a match*****')
print(euclidean(modi_face_1, modi_face_2))
print(euclidean(modi_face_1, modi_face_3))
print(euclidean(modi_face_1, biden_face_1))





         
 