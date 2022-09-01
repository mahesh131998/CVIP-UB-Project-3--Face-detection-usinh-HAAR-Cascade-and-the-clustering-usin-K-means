# CVIP-UB-Project-3--Face-detection-usinh-HAAR-Cascade-and-the-clustering-usin-K-means
run UBFaceDetection.py file
Train haar cascade and store the rsults in face_detection.xml(i have used pretrained model)
FaceDetector.py use to detect face and create bounding box around it, coordinates are stored in results.json file .
using that coordinates images are cropped and faces are clustered using FaceCluster.py using Kmeans algorithm.
