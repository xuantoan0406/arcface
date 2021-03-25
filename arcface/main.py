import faceRecognit as fr
import cv2
import timeit
start = timeit.default_timer()
X_test=fr.X_test
image=X_test[1]
#du doan

a=fr.faceRecognit(image)
stop = timeit.default_timer()
print(a)
print(stop-start)