import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('C:/Users/USER/hajunseo/CV_PA2/SfM/Data/sfm03.jpg')
img2 = cv2.imread('C:/Users/USER/hajunseo/CV_PA2/SfM/Data/sfm04.jpg')


plt.figure(1)
plt.imshow(img1)

plt.figure(2)
plt.imshow(img2)

plt.show(block=False)

plt.close('all')

sift = cv2.SIFT_create()
detect1 = sift.detect(img1, None)     ## keypoint
detect2 = sift.detect(img2, None)

img1_d = cv2.drawKeypoints(img1, detect1, None)
img2_d = cv2.drawKeypoints(img2, detect2, None)


##bf = cv2.BFMatcher()
##matches = bf.match(detect1,detect2)

plt.figure(3)
plt.imshow(img1_d)
plt.figure(4)
plt.imshow(img2_d)
plt.show(block=False)


kp1, desc1 = detector.detectAndCompute(img1, None)
kp2, desc2 = detector.detectAndCompute(img2, None)


##cv2.imshow('img1', img1_d)    #(title, image)
##cv2.waitKey()
##cv2.imshow('img2', img2_d)
##cv2.waitKey()
##cv2.destroyAllWindows()
