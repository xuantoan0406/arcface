import sys
import os
from os import listdir
import cv2
path_foder_image_raw=sys.argv[1]
id=sys.argv[2]
def crop_face(foder,id):
   if os.path.exists("./faceDetect/{}".format(id)) is False:
      os.mkdir("./faceDetect/{}".format(id))
   for name in listdir(foder):
      if id != '.DS_Store':
         image=cv2.imread(foder+"/"+name)
         cv2.imwrite("./faceDetect/{}/{}".format(id,name),image)
   return "ok"

if __name__=="__main__":
   crop_face(path_foder_image_raw,id)