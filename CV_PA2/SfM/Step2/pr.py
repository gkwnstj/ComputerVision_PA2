import matlab.engine
import numpy as np
import time
import cv2




eng = matlab.engine.start_matlab()
eng.addpath(r'C:/Users/USER/hajunseo/CV_PA2/SfM/Step2', nargout=0) # 'calibrated_fivepoint.m'가 위치한 경로
end = time.time()
for i in range(1):
    a = np.random.rand(3,5).tolist()
    a = matlab.double(a)
    print("a : ", a)
    b = np.random.rand(3, 5).tolist()
    b = matlab.double(b)
    print("b : ", b)
    E = eng.calibrated_fivepoint(a, b)
    print("E : ", np.asarray(E))
print("time : ", time.time()-end)

eng.quit()
