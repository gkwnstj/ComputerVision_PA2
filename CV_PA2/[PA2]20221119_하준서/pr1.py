import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

E = np.array([[-0.01366275,  0.05725587, -0.07659953],
       [-0.14753017, -0.02133212, -0.68699658],
       [ 0.02833085,  0.70372191, -0.02135249]])

p1 = np.array([[1179.0656, 1914.1427]])
p2 = np.array([[1179.0656, 1914.1427]])

CM = np.array([[]])
# Essential matrix to camera matrix
U, S, VT = np.linalg.svd(E, full_matrices=True)
W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])

# camera matrix = P'= [UWVT | +u3] or [UWVT | −u3] or [UWTVT | +u3] or [UWTVT | −u3].
camera_matrix = np.array([
    np.column_stack((U @ W @ VT, U[:,2])),
    np.column_stack((U @ W @ VT, -U[:,2])),
    np.column_stack((U @ W.T @ VT, U[:,2])),
    np.column_stack((U @ W.T @ VT, -U[:,2]))])

# camera pose matrix , Rt x , 3d point HOMOGENIEUOUS 
for i in range(4): 
    tmp = camera_matrix[i]
    for j in range(len(p1)):
        a = p1[j].flatten()
        print(a)
        b = p2[j].flatten()
        print(b)
        c = np.concatenate((a, b))
        print(c)
        d = tmp@c.T # 3x4@4x4
        print(d)
        if np.any(d<0):
            break
        else:
            CM = camera_matrix[i]
