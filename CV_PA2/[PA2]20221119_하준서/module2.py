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
    if m.distance < 0.999*n.distance:
        good.append([m])
        

print("good : ", len(good))  

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3)
plt.savefig("img")
plt.show(block=False)

############################### opencv documentation #######################
good = sum(good, [])
query_idx = [match.queryIdx for match in good]
train_idx = [match.trainIdx for match in good]
p1 = np.float32([kp1[ind].pt for ind in query_idx]) # coordinate (?assume normalized?)
p2 = np.float32([kp2[ind].pt for ind in train_idx])

onelist = []
for i in range(0, len(p1)):
    onelist.append([1])

onelistnp = np.array(onelist)
p1_norm = np.concatenate((p1,onelistnp),axis=1)
p2_norm = np.concatenate((p2,onelistnp),axis=1)

####################################################################################
plt.close('all')
import matlab.engine
import numpy as np
import time
import cv2

# https://kr.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
# Window/Ubuntu terminal에서 "{Matlab 설치 경로}/extern/engines/python"으로 이동 후,
# python setup.py install 실행

# https://mscipio.github.io/post/matlab-from-python/

eng = matlab.engine.start_matlab()
eng.addpath(r'C:/Users/USER/hajunseo/CV_PA2/SfM/Step2', nargout=0) # 'calibrated_fivepoint.m'가 위치한 경로
end = time.time()
inlierlist = []
for i in range(100):
    a = np.random.rand(3,5).tolist()
    a = matlab.double(a)
    b = np.random.rand(3, 5).tolist()
    b = matlab.double(b)
    E = eng.calibrated_fivepoint(a, b)
##    print("E : ", np.asarray(E))

##eng.quit()

    Essen_list = np.asarray(E)

    ess1 = Essen_list[:,0]

    ess1 = ess1.reshape(3,3)
    detess1 = np.linalg.det(ess1)

    for i in range(0,len(p1_norm)):
        
        m = p2_norm[i] * ess1 * p2_norm[i].transpose()
        if (np.sum(m) < 0.1 and np.sum(m)>-0.1):
            inlierlist.append(m)

print(inlierlist)

