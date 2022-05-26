# -*- coding: utf-8 -*-
"""
Created on Thu May 26 08:33:06 2022

@author: arti

"""

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path='ImagesAttendance'
images=[]
className=[]
myList=os.listdir(path)
print(myList)

for cl in myList:
    cur_img=cv2.imread(f'{path}/{cl}')
    images.append(cur_img)
    className.append(os.path.splitext(cl)[0])
print(className)    



def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList




#############################marking attendance##################
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name  not in nameList :
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
        
        
#markAttendance('arti')        
markAttendance('modi')   
# markAttendance('biden')        
# markAttendance('elon')  
# markAttendance('satya-nande')  






encodeListKnown=findEncodings(images)    
print('Encoding Complete')


cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    imgS=cv2.resize(img, (0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    
    facesCurFrame=face_recognition.face_locations(imgS)
    encodesCurFrame=face_recognition.face_encodings(imgS,facesCurFrame)
    
    
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex=np.argmin(faceDis)
        
        if matches[matchIndex]:
            name=className[matchIndex].upper()
            print(name)
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0),2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2),  (0,255,0),cv2.FILLED)
            cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255),2)
            
            
    cv2.imshow("WebCam", img)
    cv2.waitKey(1)
    
    

        
# face_loc=face_recognition.face_locations(imgElon)[0]
# encodeElon=face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon, (face_loc[3],face_loc[0]), (face_loc[1],face_loc[2]), (255,0,255),2)   

# face_loc_test=face_recognition.face_locations(imgTest)[0]
# encode_test=face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(face_loc[3],face_loc[0]), (face_loc[1],face_loc[2]),(255,0,255),2) 

# results=face_recognition.comapare_faces([encodeElon],encode_test) 
# face_distance=face_recognition.face_distance([encodeElon],encode_test)









#img_arti=face_recognition.load_image_file('ImagesBasics/arti.jpg')
#img_arti=cv2.cvtColor(img_arti, cv2.COLOR_BGR2RGB)

#img_biden=face_recognition.load_image_file('ImagesBasics/biden.jpg')
#img_biden=cv2.cvtColor(img_biden, cv2.COLOR_BGR2RGB)

#img_Elon_Mask=face_recognition.load_image_file('ImagesBasics/Elon-Mask.jpg')
#img_Elon_Mask=cv2.cvtColor(img_Elon_Mask, cv2.COLOR_BGR2RGB)

#img_modi=face_recognition.load_image_file('ImagesBasics/modi.jpg')
#img_modi=cv2.cvtColor(img_modi, cv2.COLOR_BGR2RGB)

#img_satya_nadella=face_recognition.load_image_file('ImagesBasics/satya-nadella.jpg')
#img_satya_nadella=cv2.cvtColor(img_satya_nadella, cv2.COLOR_BGR2RGB)


################### correct this statement###############



















