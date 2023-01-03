import matlab.engine
import numpy as np
import time
import cv2

# https://kr.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
# Window/Ubuntu terminal에서 "{Matlab 설치 경로}/extern/engines/python"으로 이동 후,
# python setup.py install 실행

# https://mscipio.github.io/post/matlab-from-python/

def main():

    eng = matlab.engine.start_matlab()
    eng.addpath(r'C:/Users/USER/hajunseo/CV_PA2/SfM/Step2', nargout=0) # 'calibrated_fivepoint.m'가 위치한 경로
    end = time.time()
    for i in range(100):
        a = np.random.rand(3,5).tolist()
        a = matlab.double(a)
##        print("a : ", a)
        b = np.random.rand(3, 5).tolist()
        b = matlab.double(b)
##        print("b : ", b)
        E = eng.calibrated_fivepoint(a, b)
        print("E : ", np.asarray(E))
        np.save("C:/Users/USER/hajunseo/CV_PA2/SfM/Step2",np.asarray(E))
##        print("E shape : ", len(E), len(E[1]))
        print(type(E))
    print("time : ", time.time()-end)

    eng.quit()


if __name__ == '__main__':
    main()

