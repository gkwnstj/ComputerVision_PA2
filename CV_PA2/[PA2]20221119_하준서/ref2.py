def randomsample(p1, p2):
    p1p2 = np.concatenate((p1, p2), axis=1)
    p1p2_ = p1p2[np.random.randint(p1p2.shape[0], size=len(p1)), :]
    p1s = p1p2_[:,:2]
    p2s = p1p2_[:,2:]
    return p1s, p2s

def RANSAC(p1, p2, iteration):
    
    b_inlier = np.array([[]]) # best inlier
    b_E = None # best Essential Matrix
    tmp_inlier_len = 0 # tmp inlier의 개수

    for i in range(iteration):
        
        # choice random sample
        p1s, p2s = randomsample(p1, p2)
        
        # 5-point algorithm (with intrinsic parameter, epipolar constraint = 0, no RANSAC) 
        cur_E, cur_inlier = cv2.findEssentialMat(p1s, p2s, focal=3092.8, pp=(2016, 1512),
                                                 maxIters = 0, threshold=0.1)

        # inlier 추출
        inlier_idx = np.where(cur_inlier==1)
        pts1 = np.vstack([p1[i] for i in inlier_idx[0]])
        pts2 = np.vstack([p2[i] for i in inlier_idx[0]])
        
        if cur_E is None:
            continue

        # inlier 수가 가장 많은 최적의 E 추출
        if len(pts1) > tmp_inlier_len: # 현재 inlier 수 > 가장 많은 inlier 수
            b_E = cur_E 
            b_inlier = cur_inlier 
            tmp_inlier_len = len(pts1)

    return b_E
