from json import load
from scipy.io import loadmat
import numpy as np
import cv2
import os

def load_data(datapath = 'DATA'):
    landmarks_path = os.path.join(datapath, "Example_FindFace_landmarks_MUCT.mat")
    face_file_names = os.path.join(datapath, "face_file_names.txt")
    face_file_dir = os.path.join(datapath, "valid_faces")

    with open(face_file_names, 'r') as f:
        face_files = f.readlines()

    face_paths = [os.path.join(face_file_dir, face_file.strip()) for face_file in face_files]

    landmarks_data = loadmat(landmarks_path)
    landmarks = landmarks_data['allLandmarks']

    return landmarks, face_paths

# def get_landmarks():
#     data = loadmat(DATA_PATH)
    
