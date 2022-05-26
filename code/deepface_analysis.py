# -*- coding: utf-8 -*-
"""
Created on Tue May 24 18:53:07 2022

@author: Lenovo
"""

from deepface import DeepFace


#face analysis

face_analysis=DeepFace.analyze(img_path='testing/modi1.jpg',actions=['emotion','age','gender','race'])


print(face_analysis)

