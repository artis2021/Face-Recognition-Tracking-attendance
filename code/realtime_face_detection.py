



import cv2
import face_recognition

#capture the video from default camera
webcam_video_stream=cv2.VideoCapture(0)

#initilise the array location to hold all face location in aframe
all_face_locations=[]

while True:
    #get the current frame from the video stream as an image
    ret,current_frame=webcam_video_stream.read()
    #resize the current frame to 1/4 size to process faster
    current_frame_small=cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    #detect all face in the images
    #arguments are image,no of time to upsample,model
    all_face_locations=face_recognition.face_locations(current_frame_small,model='hog')
    
    #loop through the face location
    for i, current_face_location in enumerate(all_face_locations):
        #splitting the tuple to get four positon valus of cuurent face
        top_pos,right_pos,buttom_pos,l_pos=current_face_location
        top_pos=top_pos*4
        right_pos=right_pos*4
        buttom_pos=buttom_pos*4
        l_pos=l_pos*4
        #printing the location of cuurent face
        print("found face {} at top: {},right:{},buttom:{},left:{}".format(i+1, top_pos,right_pos,buttom_pos,l_pos))
         #draw rectangle around the face detection
        cv2.rectangle(current_frame, (l_pos,top_pos), (right_pos,buttom_pos), (0,0,255),2)
        #showing the cuuent facce with rectangle drawn
        cv2.imshow("WebCame Video", current_frame)
    
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
    
webcam_video_stream.release()
cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    