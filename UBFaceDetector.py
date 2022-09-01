'''
All of your implementation should be in this file.
'''
'''
This is the only .py file you need to submit. 
'''
'''
    Please do not use cv2.imwrite() and cv2.imshow() in this function.
    If you want to show an image for debugging, please use show_image() function in helper.py.
    Please do not save any intermediate files in your final submission.
'''
from helper import show_image

import cv2
import numpy as np
import os
import sys
import glob
import face_recognition
from sklearn.cluster import KMeans


'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''


def detect_faces(input_path: str) -> dict:
    result_list = []
    
    '''
    Your implementation.
    '''
    face_det = cv2.CascadeClassifier('face_detection.xml')
    for filename in glob.glob(input_path+'/*'):
        head=os.path.basename(filename)
        im=cv2.imread(filename)
        im1= cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        bounding_boxes = face_det.detectMultiScale(image = im1, scaleFactor = 1.1, minNeighbors = 8)
        for i in range(len(bounding_boxes)):
            dic= dict(iname= head , bbox=[int(i) for i in list(bounding_boxes[i])])
            result_list.append(dic)
    return result_list


'''
K: number of clusters
# '''
def cluster_faces(input_path: str, K: int) -> dict:
    result_list = []
    dim_list = []
    head_list=[]
    images_list=[]
    K= int(K)
    '''
    Your implementation.
    '''
    face_det = cv2.CascadeClassifier('face_detection.xml')
    for filename in glob.glob(input_path+'/*'):
        head=os.path.basename(filename)
        head_list.append(head)
        im=cv2.imread(filename)
        im1= cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        images_list.append(im1)
        bounding_boxes = face_det.detectMultiScale(image = im1, scaleFactor = 1.1, minNeighbors = 8)
        bounding_boxes= bounding_boxes[0]
        bounding_boxes=list(bounding_boxes)
        for i in range(len(bounding_boxes)):
            bounding_boxes[i]= int(bounding_boxes[i])
        # print(bounding_boxes)
        x, y, w, h=bounding_boxes
        croped_image= im1[y:y+h, x:x+w]
        dimentional_vector= face_recognition.face_encodings(croped_image)[0]
        dim_list.append(dimentional_vector)
    
    kmeans_images = KMeans(n_clusters=K, random_state=0).fit(dim_list)
    print(kmeans_images.labels_)
    labels=kmeans_images.labels_

    

    for i in range(K):
        # print(i)
        imgs=[]
        con=[]
        for j in range(len(labels)):
            if i == labels[j]:
                imgs.append(head_list[j])

        dic= dict(cluster_no=i , elements= imgs)
        result_list.append(dic)

    print(result_list)

        

    return result_list


'''
If you want to write your implementation in multiple functions, you can write them here. 
But remember the above 2 functions are the only functions that will be called by FaceCluster.py and FaceDetector.py.
'''

"""
Your implementation of other functions (if needed).
"""
