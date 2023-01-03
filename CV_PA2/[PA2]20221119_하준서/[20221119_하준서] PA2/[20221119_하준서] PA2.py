import cv2
import matplotlib.pyplot as plt
import numpy as np

##from cv2 import xfeatures2d

img1 = cv2.imread('C:/Users/USER/hajunseo/CV_PA2/SfM/Data/sfm03.jpg')
img2 = cv2.imread('C:/Users/USER/hajunseo/CV_PA2/SfM/Data/sfm04.jpg')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)


##sift = cv2.xfeatures2d.SIFT_create()

plt.figure(1)
plt.imshow(img1)

plt.figure(3)
plt.imshow(img2)

plt.show(block=False)


############################### opencv documentation #######################

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)


bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)


print("matches : ", len(matches))


# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 1*n.distance:     # 0.97 => 3048
        good.append([m])
print("good : ", len(good))  



img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3)
plt.savefig("img")
plt.show(block=False)

#########################################################################

############################### Essential matrix #########################
good = sum(good, [])
query_idx = [match.queryIdx for match in good]
train_idx = [match.trainIdx for match in good]
p1 = np.float32([kp1[ind].pt for ind in query_idx]) # coordinate (?assume normalized?)
p2 = np.float32([kp2[ind].pt for ind in train_idx])

############################## normalization ###########################

intrinsic = np.matrix([[3092.8, 0.0, 2016],
                       [0.0, 3092.8, 1512],
                       [0.0, 0.0, 1.0]])
##np.linalg.inv(intrinsic)
##norm_coord = np.linalg.inv(intrinsic) * p1

normp1x = (p1[:,0]-2016)/3092.8
normp1x = normp1x.reshape(len(normp1x),1)
normp1y = (p1[:,1]-1512)/3092.8
normp1y = normp1y.reshape(len(normp1y),1)

normp2x = (p2[:,0]-2016)/3092.8
normp2x = normp2x.reshape(len(normp2x),1)
normp2y = (p2[:,1]-1512)/3092.8
normp2y = normp2y.reshape(len(normp2y),1)

norm_p1 = np.concatenate((normp1x,normp1y),axis=1)
norm_p2 = np.concatenate((normp2x,normp2y),axis=1)

onelist = []
for i in range(0, len(norm_p1)):
    onelist.append([1])

onelistnp = np.array(onelist)
p1_norm = np.concatenate((norm_p1,onelistnp),axis=1)
p2_norm = np.concatenate((norm_p2,onelistnp),axis=1)


#########################################################################

##E, mask = cv2.findEssentialMat(p1, p2, method=cv2.RANSAC, focal=3092.8, pp=(2016, 1512), maxIters = 2000, threshold=0.0005)
##
##p1_inlier = p1[mask.ravel()==1] # left image inlier
##p2_inlier= p2[mask.ravel()==1] # right image inlier

######################################################################

##plt.close('all')
####################### Essenstial matrix with RANSAC ##############################
import matlab.engine
import numpy as np
import time
import cv2

eng = matlab.engine.start_matlab()
eng.addpath(r'...', nargout=0) # 'calibrated_fivepoint.m'가 위치한 경로
end = time.time()

idxlist = []
inlierlist= []
Esslist = []
for i in range(500):
    ########## Choosing 5-point ##########
    for j in range(5):
        idx = np.random.randint(len(p1_norm)) 
        idxlist.append(idx) 
    ######################################
    a = p1_norm[idxlist[:]].tolist()    ### 5 point
    b = p2_norm[idxlist[:]].tolist()    ### 5 point
    a1 = np.array(a).T
    b1 = np.array(b).T
    a = matlab.double(a1)
    b = matlab.double(b1)
    E = eng.calibrated_fivepoint(a, b)
    Essen_list = np.asarray(E)   ########### ??? transpose?
    idxlist.clear()

    for h in range(Essen_list.shape[1]):
        ess1 = Essen_list[:,h]
        ess1 = ess1.reshape(3,3)
        
        if np.all(np.diag(a1.T@ess1@b1)<0.0005) and np.all(np.diag(a1.T@ess1@b1)>-0.0005):
            Esslist.append(ess1)
    
##    np.diag(a.T@Esslist[3]@b)
    

ulist = []
ranidxlist = []
ranidxlist1 = []
for k in range(len(Esslist)):
    ess2 = Esslist[k]
    u = 0
    ranidxlist1.clear()
    for l in range(0,2000):
        ranidx = np.random.randint(len(p1))
        m = np.matrix(p2_norm)[ranidx] * np.matrix(ess2) * np.matrix(p1_norm)[ranidx].transpose()
        if (m < 0.005): 
            ranidxlist1.append(ranidx)
            u = u + 1
    
    ranidxlist.append(ranidxlist1)
    ulist.append(u)
    print("number of inlier : ", u)

E = Esslist[ulist.index(max(ulist))]

p1in = []
p2in = []
for i in range(len(ranidxlist[ulist.index(max(ulist))])):
    inlieridx = ranidxlist[ulist.index(max(ulist))][i]
    p1in.append(p1_norm[inlieridx])
    p2in.append(p2_norm[inlieridx])
    
    
p1_normlin = np.array(p1in)
p2_normlin = np.array(p2in)


######################## Essential Matrix decomposition ###################
U, S, VT = np.linalg.svd(E)

W = np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]])

P1 = np.concatenate( (np.matrix(U)*np.matrix(W)*np.matrix(VT), np.matrix(U[:,2]).transpose()), axis = 1 )
P2 = np.concatenate( (np.matrix(U)*np.matrix(W)*np.matrix(VT), np.matrix(-U[:,2]).transpose()), axis = 1 )
P3 = np.concatenate( (np.matrix(U)*np.matrix(W).transpose()*np.matrix(VT), np.matrix(U[:,2]).transpose()), axis = 1 )
P4 = np.concatenate( (np.matrix(U)*np.matrix(W).transpose()*np.matrix(VT), np.matrix(-U[:,2]).transpose()), axis = 1 )

Plist = [P1,P2,P3,P4]

P_1 = np.concatenate((np.matrix(np.eye(3)),np.matrix([0,0,0]).transpose()), axis = 1)



P_2 = None

for i in range(4): 
    tmp = Plist[i]
    for j in range(len(p1)):
        a = p1[j,0:2].flatten()
        b = p2[j,0:2].flatten()
        c = np.concatenate((a, b))
        d = tmp@c.T 
        if np.any(d<0):
            break
        else:
            P_2 = Plist[i]

if P_2 is None:
    P_2 = Plist[0]
    print("some points are behind the camera")



pt1 = P_1[0,:]
pt2 = P_1[1,:]
pt3 = P_1[2,:]
pt11 = P_2[0,:]
pt22 = P_2[1,:]
pt33 = P_2[2,:]

pointslist = []

for s in range(len(p1_normlin)):
    

    A = np.array([p1_normlin[s][0]*pt3 - pt1,
                 p1_normlin[s][1]*pt3 - pt2,
                 p2_normlin[s][0]*pt33 - pt11,
                 p2_normlin[s][1]*pt33 - pt22])

    MatA = np.matrix(A)
    U, S, V = np.linalg.svd(MatA)

    points = V[:-1,3]


    pointslist.append(np.array(points).reshape(-1))

points3d = np.array(pointslist)


x = points3d[:,0]
y = points3d[:,1]
z = points3d[:,2]

plt.close('all')

fig = plt.figure(figsize=(15,15))
ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z, c='b', marker='o') 
plt.show(block=False)



import numpy as np
import pandas as pd

from pyntcloud import PyntCloud

cloud = PyntCloud(pd.DataFrame(
    # same arguments that you are passing to visualize_pcl
    data=np.concatenate((x.reshape(len(x),1),y.reshape(len(x),1),z.reshape(len(x),1)), axis = 1),
    columns=["x", "y", "z"]))

cloud.to_file("output.ply")


