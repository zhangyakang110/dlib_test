#coding=utf-8
import dlib
import sys
import os
import pdb
from skimage import io
import re
import numpy as np

now_path = os.path.abspath(__file__)

base_dir = os.path.dirname(now_path)

all_face_path = os.path.join(base_dir,"all_face")

model_name = "shape_predictor_68_face_landmarks.dat"#"shape_predictor_5_face_landmarks.dat"

five_model_path = os.path.join(base_dir,'model',model_name)

detector = dlib.get_frontal_face_detector()
make_pointer = dlib.shape_predictor(five_model_path)

def get_all_imgs():
    all_imgpaths = [os.path.join(all_face_path,imgname) for imgname in os.listdir(all_face_path)]
    return all_imgpaths

def get_points(imgpath):
    img = io.imread(imgpath)
    dets = detector(img,1)
    for index,face in enumerate(dets):
        #pdb.set_trace()
        shape = make_pointer(img,face)
       	five_points = [(shape.part(19).x,shape.part(36).y),(shape.part(23).x,shape.part(45).y),(shape.part(30).x,shape.part(30).y),(shape.part(48).x,shape.part(48).y),(shape.part(54).x,shape.part(54).y)]
        print(five_points) 
        
def main():
    all_imgs = get_all_imgs()
    for imgpath in all_imgs:
        print(imgpath)
        get_points(imgpath)

if __name__ == "__main__":
    main()
