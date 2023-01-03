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


bf = cv2.BFMatcher(cv2.NORM_L2)
##matches = bf.match(des1,des2)
matches = bf.knnMatch(des1,des2,k=2)

print("matches : ", len(matches))

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.8*n.distance:     # 0.97 => 3048
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
normp1y = normp1x.reshape(len(normp1y),1)

norm_p1 = np.concatenate((normp1x,normp1y),axis=1)

normp2x = (p1[:,0]-2016)/3092.8
normp2x = normp2x.reshape(len(normp2x),1)
normp2y = (p1[:,1]-1512)/3092.8
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

plt.close('all')
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

    for k in range(len(Essen_list.transpose())):
        ess1 = Essen_list[:,k]
        ess1 = ess1.reshape(3,3)
        
        for l in range(0,len(p1_norm)):
            
            m = np.matrix(p2_norm)[l] * np.matrix(ess1) * np.matrix(p1_norm)[l].transpose()
            if (m <= 0.0005): #  and np.sum(m)>-0.1):
                inlierlist.append(m)
##                esslist.append(ess1)


print(Essen_list)

##p1_norm[aidxlist[:]]



####################### Essential matrix Decomposition #################





