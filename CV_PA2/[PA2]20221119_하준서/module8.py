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

plt.figure(2)
plt.imshow(img2)

plt.show(block=False)

plt.close('all')

############################### opencv documentation #######################

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)


bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
##matches = bf.match(des1, des2)



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

intrinsic = np.matrix([[3092.8, 0.0, 2016], [0.0, 3092.8, 1512], [0.0, 0.0, 1.0]])
##np.linalg.inv(intrinsic)
##norm_coord = np.linalg.inv(intrinsic) * p1

normp1x = (p1[:,0]-2016)/3092.8
normp1x = normp1x.reshape(len(normp1x),1)
normp1y = (p1[:,1]-1512)/3092.8
normp1y = normp1y.reshape(len(normp1y),1)

norm_p1 = np.concatenate((normp1x,normp1y),axis=1)

normp2x = (p2[:,0]-2016)/3092.8
normp2x = normp2x.reshape(len(normp2x),1)
normp2y = (p2[:,1]-1512)/3092.8
normp2y = normp2y.reshape(len(normp2y),1)

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
eng.addpath(r'C:/Users/USER/hajunseo/CV_PA2/SfM/Step2', nargout=0) # 'calibrated_fivepoint.m'가 위치한 경로
end = time.time()

aidxlist = []
bidxlist = []
inlierlist= [] 
for i in range(1):
    ########## Choosing 5-point ##########
    for j in range(5):
        aidx = np.random.randint(len(p1))
        bidx = np.random.randint(len(p1))
        aidxlist.append(aidx)
        bidxlist.append(bidx)
    ######################################
    a = p1_norm[aidxlist[:]].tolist()    ### 5 point
    b = p2_norm[bidxlist[:]].tolist()    ### 5 point
    a = matlab.double(a)
    b = matlab.double(b)
    E = eng.calibrated_fivepoint(a, b)
    Essen_list = np.asarray(E)

Esslist = []
ulist = []
for k in range(len(Essen_list.transpose())):
    ess1 = Essen_list[:,k]
    ess1 = ess1.reshape(3,3)
    Esslist.append(ess1)
    u = 0
    for l in range(0,2000):
        
        m = np.matrix(p2_norm)[np.random.randint(len(p1))] * np.matrix(ess1) * np.matrix(p1_norm)[np.random.randint(len(p2))].transpose()
        if (m <= 0.0005): #  and np.sum(m)>-0.1):
            u = u + 1

    ulist.append(u)
    print("number of inlier : ", u)

E = Esslist[ulist.index(max(ulist))]


######################## Essential Matrix decomposition ###################
U, S, VT = np.linalg.svd(E)

W = np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]])

P1 = np.concatenate( (np.matrix(U)*np.matrix(W)*np.matrix(VT), np.matrix(U[:,2]).transpose()), axis = 1 )
P2 = np.concatenate( (np.matrix(U)*np.matrix(W)*np.matrix(VT), np.matrix(-U[:,2]).transpose()), axis = 1 )
P3 = np.concatenate( (np.matrix(U)*np.matrix(W).transpose()*np.matrix(VT), np.matrix(U[:,2]).transpose()), axis = 1 )
P4 = np.concatenate( (np.matrix(U)*np.matrix(W).transpose()*np.matrix(VT), np.matrix(-U[:,2]).transpose()), axis = 1 )

Plist = [P1,P2,P3,P4]

P_1 = np.concatenate((np.matrix(np.eye(3)),np.matrix([0,0,0]).transpose()), axis = 1)


for i in range(4): 
    tmp = Plist[i]
    for j in range(len(p1)):
        a = p1[j].flatten()
        b = p2[j].flatten()
        c = np.concatenate((a, b))
        d = tmp@c.T # 3x4@4x4
        if np.any(d<0):
            break
        else:
            P_2 = Plist[i]



    
pt1 = P_1[0,:]
pt2 = P_1[1,:]
pt3 = P_1[2,:]
pt11 = P_2[0,:]
pt22 = P_2[1,:]
pt33 = P_2[2,:]

pointslist = []

for s in range(len(p1_norm)):
    

    A = np.array([p1_norm[s][0]*pt3 - pt1,
                 p1_norm[s][1]*pt3 - pt2,
                 p2_norm[s][0]*pt33 - pt11,
                 p2_norm[s][1]*pt33 - pt22])

    MatA = np.matrix(A)

    U, S, V = np.linalg.svd(MatA)
 
    points = V[:-1,3]/V[3,3]
##    points = V[3,0:3]/V[3,3]
    

    pointslist.append(np.array(points).reshape(-1))

points3d = np.array(pointslist)

x = points3d[:,0]
y = points3d[:,1]
z = points3d[:,2]


fig = plt.figure(figsize=(15,15))
ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z, c='b', marker='o') 
plt.show()
