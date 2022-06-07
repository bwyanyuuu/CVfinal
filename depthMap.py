import numpy as np
import cv2
from scipy.sparse import spdiags

# def wlsFilter(img, Lambda, alpha=1.2, eps=0.0001):
    
#     L = np.log(img+0.0001)
#     row, cols = img.shape[:2]
#     k = row * cols
    
#     #對L矩陣的第一維度上做差分，也就是下面的行減去上面的行，得到(N-1)xM維的矩陣
#     dy = np.diff(L, 1, 0)
#     dy = -Lambda/(np.power(np.abs(dy),alpha)+eps)
#     #在最後一行的後面補上一行0
#     dy = np.pad(dy, ((0,1),(0,0)), 'constant')
#     #按列生成向量，對應上面Ay的對角線元素
#     dy = dy.T
#     dy = dy.reshape(-1,1)
    
#     #對L矩陣的第二維度上做差分，也就是右邊的列減去左邊的列，得到Nx(M-1)的矩陣
#     dx = np.diff(L, 1, 1)
#     dx = -Lambda/(np.power(np.abs(dx),alpha)+eps)
#     #在最後一列的後面補上一行0
#     dx = np.pad(dx, ((0,0),(0,1)), 'constant')
#     #按列生成向量，對應上面Ay的對角線元素
#     dx = dx.T
#     dx = dy.reshape(-1,1)
    
#     #構造五點空間非齊次拉普拉斯矩陣
#     B = np.hstack((dx,dy))
#     B = B.T
#     diags = np.array([-row, -1])
#     #把dx放在-row對應的對角線上，把dy放在-1對應的對角線上
#     A = spdiags(B, diags, k, k).toarray()
    
#     e = dx
#     w = np.pad(dx, ((row,0),(0,0)), 'constant')
#     w = w[0:-row]
    
#     s = dy
#     n = np.pad(dy, ((1,0),(0,0)), 'constant')
#     n = n[0:-1]
    
#     D = 1-(e+w+s+n)
#     D = D.T
#     #A只有五個對角線上有非0元素
#     diags1 = np.array([0])
#     A = A + np.array(A).T + spdiags(D, diags1, k, k).toarray()
    
    
#     im = np.array(img)
#     p,q = im.shape[:2]
#     g = p*q
#     im = np.reshape(im,(g,1))
    
#     a = np.linalg.inv(A)
    
#     out = np.dot(a,im)
    
#     out = np.reshape(out,(row, cols))
    
#     return out

######## BM ########
# Load the left and right images in gray scale
imgLeft = cv2.imread('./img/tsukuba_l.png', 0)
imgRight = cv2.imread('./img/tsukuba_r.png', 0)

# # Initialize the stereo block matching object 
# stereo = cv2.StereoBM_create(numDisparities=32, blockSize=13)

# # Compute the disparity image
# disparity = stereo.compute(imgLeft, imgRight)

# # Normalize the image for representation
# min = disparity.min()
# max = disparity.max()
# disparity = np.uint8(255 * (disparity - min) / (max - min))
# # recover = wlsFilter(disparity, 1)

# # Display the result
# # cv2.imshow('disparity', np.hstack((imgLeft, imgRight, disparity)))
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# cv2.imwrite('./depth map/depthMap.png', disparity)

# wsize=31
# max_disp = 128
# sigma = 1.5
# lmbda = 600.0
# left_matcher = cv2.StereoBM_create(max_disp, wsize)
# right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
# left_disp = left_matcher.compute(imgLeft, imgRight)
# right_disp = right_matcher.compute(imgLeft,imgRight)

# # Now create DisparityWLSFilter
# wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
# wls_filter.setLambda(lmbda)
# wls_filter.setSigmaColor(sigma)
# filtered_disp = wls_filter.filter(left_disp, imgLeft, disparity_map_right=right_disp)
# cv2.imwrite('./depth map/filterDepthMap1.png', filtered_disp)



# #setting filter parameters
# lmbda = 80000
# sigma = 1.2
# visual_multiplier = 1.0
# wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereoMatcher)
# wls_filter.setLambda(lmbda)
# wls_filter.setSigmaColor(sigma)

# # # Using the WLS filter
# np.uint8(dispL)
# filteredImg= wls_filter.filter(dispL,grayLeft,None,dispR)
# filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=1, alpha=255, norm_type=cv2.NORM_MINMAX);

mindisparity = 0
ndisparities = 64
SADWindowSize = 5
P1 = 8 * 3 * SADWindowSize* SADWindowSize
P2 = 32 * 3 * SADWindowSize* SADWindowSize
blockSize = 5

win_size = 2
min_disp = -4
max_disp = 9
num_disp = max_disp - min_disp  # Needs to be divisible by 16
stereoSGBM = cv2.StereoSGBM_create(
    # minDisparity=min_disp,
    # numDisparities=num_disp,
    # blockSize=5,
    # uniquenessRatio=5,
    # speckleWindowSize=5,
    # speckleRange=5,
    # disp12MaxDiff=2,
    # P1=8 * 3 * win_size ** 2,
    # P2=32 * 3 * win_size ** 2,
    minDisparity=mindisparity,
    numDisparities=ndisparities,
    blockSize=SADWindowSize,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    disp12MaxDiff=1,
    P1=8 * 3 * SADWindowSize ** 2,
    P2=32 * 3 * SADWindowSize ** 2,
    mode=cv2.StereoSGBM_MODE_HH,
)
# wsize=31
# max_disp = 128
# sigma = 1.5
# lmbda = 8000.0
# left_matcher = cv2.StereoBM_create(max_disp, wsize)
# right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
# left_disp = left_matcher.compute(imgLeft, imgRight)
# right_disp = right_matcher.compute(imgLeft,imgRight)

# # Now create DisparityWLSFilter
# wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
# wls_filter.setLambda(lmbda)
# wls_filter.setSigmaColor(sigma)
# filtered_disp = wls_filter.filter(left_disp, imgLeft, disparity_map_right=right_disp)


disparity_SGBM = stereoSGBM.compute(imgLeft, imgRight)
disparity_SGBM = np.float32(disparity_SGBM/16)
min = disparity_SGBM.min()
max = disparity_SGBM.max()
print(min, max)
disparity = np.uint8(255 * (disparity_SGBM - min) / (max - min))
cv2.normalize(disparity_SGBM, disparity_SGBM, )
cv2.imwrite('./depth map/DepthMapSBGM.png', disparity)
# wls = cv2.ximgproc.createDisparityWLSFilter(stereoSGBM)
# wls.setLambda(8000)
# wls.setSigmaColor(1.5)
# filtered_disparity_map = wls.filter(disparity_SGBM, imgLeft)
# cv2.imwrite('./result/filterDepthMap.png', filtered_disparity_map)
