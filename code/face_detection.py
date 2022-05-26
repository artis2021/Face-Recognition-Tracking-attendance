



import cv2
import face_recognition

image_to_detect=cv2.imread('arti.jpg')
#cv2.imshow("test", image_to_detect)

all_face_locations=face_recognition.face_locations(image_to_detect,model="hog")
print("there are {} no of faces in given image".format(len(all_face_locations)))

for i, current_face_location in enumerate(all_face_locations):
    #splitting the tuple to get four positon valus of cuurent face
    top_pos,right_pos,buttom_pos,l_pos = current_face_location
    #printing the location of cuurent face
    print("found face {} at top: {},right:{},buttom:{},left:{}".format(i+1, top_pos,right_pos,buttom_pos,l_pos))
    current_face_image=image_to_detect[top_pos:buttom_pos,l_pos:right_pos]
    cv2.imshow("face no "+str(i+1),current_face_image)
    cv2.waitKey(0)