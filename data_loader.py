from json import load
from scipy.io import loadmat
import numpy as np

DATA_PATH = "/home/PJLAB/weigengchen/CodeSpace/projects/ASM/ActiveShapeModels/Landmarks/Example_FindFace_landmarks_MUCT.mat"

def get_landmarks():
    data = loadmat(DATA_PATH)
    