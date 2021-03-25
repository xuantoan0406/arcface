import cv2
import timeit
import numpy as np
from matplotlib import pyplot as plt
#from arcface import ArcFace
import file_def as fd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

start = timeit.default_timer()
faces = fd._load_pickle("./data/faces.pkl")
names = fd._load_pickle("data/names.pkl")

stop = timeit.default_timer()
print(stop-start)
'''
X_train, X_test, y_train, y_test = train_test_split(faces, names, test_size=0.25, stratify=names)

# tim nhan dan gan nhat
def _most_similarity(embed_vecs, vec, labels):
    sim = cosine_similarity(embed_vecs, vec)
    sim = np.squeeze(sim, axis=1)
    argmax = np.argsort(sim)[::-1][:1]
    if sim[argmax[0]] < 0.4:
        label = "unknown"
    else:
        label = [labels[idx] for idx in argmax][0]
    return label

# tach anh thanh vector 512 chieu
face_rec = ArcFace.ArcFace()
X_train_vec = face_rec.calc_emb(X_train)
X_test_vec = face_rec.calc_emb(X_test)

# danh gia mo hinh
def accuracy():
    y_preds = []
    for vec in X_test_vec:
        vec = vec.reshape(1, -1)
        y_pred = _most_similarity(X_train_vec, vec, y_train)
        y_preds.append(y_pred)
    accuracies = accuracy_score(y_preds, y_test)
    return accuracies
def faceRecognit(image):
    face=cv2.resize(image,(112,112))
    vec=face_rec.calc_emb(face).reshape(1,-1)
    name=_most_similarity(X_train_vec,vec,y_train)
    return name
'''
