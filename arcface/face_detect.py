import insightface
import os
from os import listdir
import cv2
import numpy as np

model = insightface.model_zoo.get_model('retinaface_r50_v1')
model.prepare(ctx_id=-1, nms=0.4)

def face_dectection(raw_folder):
   print("start processing...")
   count_image = 0
   for name in listdir(raw_folder):
      if name != '.DS_Store':
         index = 0
         os.mkdir("./faceDetect/{}".format(name))
         for image in listdir(raw_folder + name):
            if image != '.DS_Store':
               image = cv2.imread(raw_folder + name + "/" + image)
               bbox, landmark = model.detect(image, threshold=0.5, scale=1.0)
               if len(bbox) != 0:
                  face_detect = image[abs(int(bbox[0][1])):int(bbox[0][3]),
                                abs(int(bbox[0][0])):int(bbox[0][2])].copy()
                  resized = cv2.resize(face_detect, (112, 112))
                  cv2.imwrite("./faceDetect/{}/{}.png".format(name, index), resized)
                  index += 1
                  count_image += 1

   print("processing success..!")
   return count_image


faceDetect_folder = "./faceDetect/"


def faces(faceDetect_folder):
   faces = []
   names = []
   for name in listdir(faceDetect_folder):
      if name != '.DS_Store':
         for face in listdir(faceDetect_folder + name):
            if face != '.DS_Store':
               face = cv2.imread(faceDetect_folder + name + '/' + face)
               faces.append(face)
               names.append(name)
   return faces, names
