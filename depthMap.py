from matplotlib.pyplot import axis
import numpy as np
import cv2
from scipy.sparse import spdiags

def normalize(mat):
    min = mat.min()
    max = mat.max()
    mat = np.uint8(255 * (mat - min) / (max - min))
    return mat.copy()
def outlier_detection(label, label_r):
    outlier = np.empty_like(label)
    # 0: not an outlier, 1: mismatch point, 2: occlusion point
    for y in range(label.shape[1]):
        for x in range(label.shape[1]):
            if x - label[y][x] < 0:
                outlier[y][x] = 2
            elif abs(label[y][x] - label_r[y][x - label[y][x]]) < 1.1:
                outlier[y][x] = 0
            else:
                for disp in range(16):
                    if x - disp > 0 and abs(disp - label_r[y][x - disp]) < 1.1:
                        outlier[y][x] = 1
                        break
                    else:
                        outlier[y][x] = 2
    return outlier


# load images
# imgLeft = cv2.imread('./img/tsukuba_l.png', 0)
# imgRight = cv2.imread('./img/tsukuba_r.png', 0)
# rgbLeft = cv2.imread('./img/tsukuba_rgb_l.png')
# rgbRight = cv2.imread('./img/tsukuba_rgb_r.png')
rgbLeft = cv2.imread('./img/im4.png')
rgbRight = cv2.imread('./img/im6.png')
rgbLeft = np.array(rgbLeft, dtype=np.int16)
rgbRight = np.array(rgbRight, dtype=np.int16)
h, w, ch = rgbLeft.shape


######## BM ########
# Initialize the stereo block matching object 
stereo_l = cv2.StereoBM_create(numDisparities=16, blockSize=13)
stereo_r = cv2.StereoBM_create(numDisparities=16, blockSize=13)
# stereo_r = cv2.ximgproc.createRightMatcher(stereo_l);

# Compute the disparity image
# dspLeft = stereo_l.compute(imgLeft, imgRight)
# dspRight = np.flip(stereo_r.compute(np.flip(imgRight, axis=1), np.flip(imgLeft, axis=1)), axis=1)

# dspLeft -=(dspLeft.min() - dspRight.min())
# dspRight = np.int8(dspRight / 32)
# dspRight = -1 * dspRight

# Normalize the image for representation
# dspLeftv = normalize(dspLeft)
# dspRightv = normalize(dspRight)

# dspRightv -= 255
# cv2.normalize(dspLeft, dspLeftv, 0, 255, cv2.NORM_MINMAX)
# cv2.normalize(dspRight, dspRightv, 0, 255, cv2.NORM_MINMAX)
# min = disparity.min()
# max = disparity.max()
# disparity = np.uint8(255 * (disparity - min) / (max - min))
# for i in range(rgbLeft.shape[0]):
#     for j in range(rgbLeft.shape[1]):
#         print(dspLeft[i][j], dspRight[i][j])
# print(dspLeft.min())
# print(dspRight.min())
# print(dspLeft.max())
# print(dspRight.max())

# dspLeft = np.int8(dspLeft / 16)
# dspRight = np.int8(dspRight / 16)

# Display the result
# cv2.imwrite('./depth map/depthMap-left.png', dspLeftv)
# cv2.imwrite('./depth map/depthMap-right.png', dspRightv)


######## SGBM ########
# mindisparity = 0
# ndisparities = 64
# SADWindowSize = 5
# P1 = 8 * 3 * SADWindowSize* SADWindowSize
# P2 = 32 * 3 * SADWindowSize* SADWindowSize
# blockSize = 5
# stereoSGBM = cv2.StereoSGBM_create(
#     minDisparity=mindisparity,
#     numDisparities=ndisparities,
#     blockSize=SADWindowSize,
#     uniquenessRatio=10,
#     speckleWindowSize=100,
#     speckleRange=2,
#     disp12MaxDiff=1,
#     P1=8 * 3 * SADWindowSize ** 2,
#     P2=32 * 3 * SADWindowSize ** 2,
#     mode=cv2.StereoSGBM_MODE_HH,
# )
# disparity_SGBM = stereoSGBM.compute(imgLeft, imgRight)
# disparity_SGBM = np.float32(disparity_SGBM/16)
# min = disparity_SGBM.min()
# max = disparity_SGBM.max()
# print(min, max)
# disparity = np.uint8(255 * (disparity_SGBM - min) / (max - min))
# cv2.normalize(disparity_SGBM, disparity_SGBM, )
# cv2.imwrite('./depth map/depthMap-SBGM.png', disparity)


######## Cost Volume Filtering ########
print('* Cost Volume Filtering')
max_disp = 192
def costVolume(rgbLeft, rgbRight, max_disp):
    print('* Cost Volume Filtering')
    h, w, ch = rgbLeft.shape
    vol = np.zeros((max_disp, h, w), dtype=np.int16)
    dis = np.zeros((h, w), dtype=np.int16)
    for i in range(rgbLeft.shape[0]):
        for j in range(rgbLeft.shape[1]):
            minNorm = 10e6
            disp = -1
            for k in range(max_disp):
                if j - k >= 0:
                    # vol[k][i][j] = np.linalg.norm(rgbLeft[i][j] - rgbRight[i][j - k])
                    vol[k][i][j] = np.mean(abs(rgbLeft[i][j] - rgbRight[i][j - k]))
                else:
                    vol[k][i][j] = np.mean(abs(rgbLeft[i][j]))
    for i in range(max_disp):
        vol[i] = cv2.bilateralFilter(normalize(vol[i]), 10, 12, 12)

    for i in range(rgbLeft.shape[0]):
        for j in range(rgbLeft.shape[1]):
            minNorm = 10e6
            disp = -1
            for k in range(max_disp):
                if vol[k][i][j] < minNorm:
                    minNorm = vol[k][i][j]
                    disp = k
            dis[i][j] = disp
    return dis.copy()
# dspLeft = costVolume(rgbLeft, rgbRight, max_disp)
# dspRight = np.flip(costVolume(np.flip(rgbRight, axis=1), np.flip(rgbLeft, axis=1), max_disp), axis=1)
# cv2.imwrite('./depth map/depthMap_l.png', normalize(dspLeft))
# cv2.imwrite('./depth map/depthMap_r.png', normalize(dspRight))
# cv2.imwrite('./depth map/l.png', np.flip(rgbLeft, axis=1))
# cv2.imwrite('./depth map/r.png', np.flip(rgbRight, axis=1))


######## cvf 2 ########
r = 9
eps = 0.0001
thresColor = 7/255
thresGrad = 2/255
gamma = 0.11
threshBorder = 3/255
gamma_c = 0.1
gamma_d = 9
r_median = 19

def matlabBGR2gray(img):
    h, w, c = img.shape
    ans = img[:, :, 0] * 0.114 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.2989
    return ans
def cvf(Il, Ir, fx_l, fx_r, max_disp):
    dispVol = np.ones((h,w,max_disp))*threshBorder
    for d in range(1,max_disp+1):
        # Right to left
        tmp = np.ones((h,w,ch))*threshBorder
        tmp[:,d:w,:] = Ir[:,:w-d,:]
        p_color = abs(tmp - Il)
        p_color = np.mean(p_color, axis = 2)
        p_color = np.minimum(p_color, thresColor)

        tmp = np.ones((h,w))*threshBorder
        tmp[:,d:w] = fx_r[:,:w-d]
        p_grad = abs(tmp - fx_l)
        p_grad = np.minimum(p_grad, thresGrad)

        p = gamma*p_color+(1-gamma)*p_grad
        dispVol[:,:,d-1] = p
    # for d in range(1,max_disp+1):
    #     cv2.imwrite('./depth map/vol%2d.png'% d, normalize(dispVol[:,:,d-1]))
    # for d in range(1,max_disp+1):
    #     dispVol[:,:,d-1] = cv2.bilateralFilter(normalize(dispVol[:,:,d-1]), 7, 12, 12).copy()
    #     # dispVol[:,:,d-1] = cv2.medianBlur(normalize(dispVol[:,:,d-1]), 3).copy()
    #     cv2.imwrite('./depth map/filvol%2d.png'% d, normalize(dispVol[:,:,d-1]))
    for d in range(max_disp):
        p = dispVol[:,:,d]

        q = cv2.ximgproc.guidedFilter(normalize(Il), normalize(p), r, eps)
        q = cv2.medianBlur(q, 5).copy()
        dispVol[:,:,d] = q
    dis = (np.argmin(dispVol, axis = 2) + np.ones((h,w)) ).astype(int)
    return dis.copy()

# Il_g = rgbLeft/255.0
# Ir_g = imgRight/255.0
Il_g = matlabBGR2gray(rgbLeft)/255.0
Ir_g = matlabBGR2gray(rgbRight)/255.0
Il = rgbLeft/255.0
Ir = rgbRight/255.0
fx_l = np.gradient(Il_g, axis=1) + 0.5
fx_r = np.gradient(Ir_g, axis=1) + 0.5
dspLeft = cvf(Il, Ir, fx_l, fx_r, max_disp)
dspRight = np.fliplr(cvf(np.fliplr(Ir).copy(), np.fliplr(Il).copy(), np.fliplr(fx_r).copy(), np.fliplr(fx_l).copy(), max_disp)).copy()
cv2.imwrite('./depth map/depthMap_l.png', normalize(dspLeft))
cv2.imwrite('./depth map/depthMap_r.png', normalize(dspRight))


######## Disparity Consistancy ########
print('* Disparity Consistancy')
print(dspLeft.shape)
check = np.zeros((h, w))
for i in range(rgbLeft.shape[0]):
    for j in range(rgbLeft.shape[1]):
        # print(dspLeft[i][j])
        # if dspLeft[i][j] == -1:
        #     continue
        if j - dspLeft[i][j] >= rgbLeft.shape[1] or j - dspLeft[i][j] < 0:
            continue
        if abs(dspLeft[i][j] - dspRight[i][j - dspLeft[i][j]]) < 2:
            check[i][j] = 255
        # print(j + dspLeft[i][j])
        # if j + dspLeft[i][j] >= rgbLeft.shape[1]:
        #     # print("**")
        #     continue
        # elif imgRight[i][j + dspLeft[i][j]] - rgbLeft[i][j] < 30:
        #     check[i][j] == 255
check = np.uint8(check)
cv2.imwrite('./depth map/depthMap-check.png', check)
# out = outlier_detection(dspLeft, dspRight)
# cv2.imwrite('./depth map/depthMap-outlier.png', out)



####### Hole Filling ########
print('* Hole Filling')
# print(dspLeft)
# print(check)
after_fill = np.zeros((h, w))
for i in range(rgbLeft.shape[0]):
    for j in range(rgbLeft.shape[1]):
        if check[i][j] == 0:
            # left
            cj = j
            fl = 10e6
            while True:
                cj -= 1
                if cj < 0:
                    break
                if check[i][cj] == 0:
                    continue
                else:
                    fl = dspLeft[i][cj]
                    break
            # right
            cj = j
            fr = 10e6
            while True:
                cj += 1
                if cj >= rgbLeft.shape[1]:
                    break
                if check[i][cj] == 0:
                    continue
                else:
                    fr = dspLeft[i][cj]
                    break
            after_fill[i][j] = min(fl, fr)
        else:
            after_fill[i][j] = dspLeft[i][j]
cv2.imwrite('./depth map/depthMap-fill.png', normalize(after_fill))


####### Weighted Medium Filter ########
# after_medium = cv2.medianBlur(after_fill, 3)
# window_size = 5
# # for i in range(rgbLeft.shape[0]):
# #     for j in range(rgbLeft.shape[1]):

# cv2.imwrite('./depth map/depthMap-medium.png', after_medium)
from scipy.signal import medfilt
from scipy.sparse import coo_matrix
import math
import numpy.matlib
def fillPixelaReference(Il, final_labels, max_disp):
    h,w = final_labels.shape
    occPix = np.zeros((h,w))
    occPix[final_labels<0] = 1
    
    fillVals = np.ones((h)) * max_disp
    final_labels_filled = final_labels.copy()

    for col in range(w):
        curCol = final_labels[:,col].copy()
        curCol[curCol==-1] = fillVals[curCol==-1]
        fillVals[curCol!=-1] = curCol[curCol!=-1]
        final_labels_filled[:,col] = curCol
    
    fillVals = np.ones((h)) * max_disp
    final_labels_filled1 = final_labels.copy()
    for col in reversed(range(w)):
        curCol = final_labels[:,col].copy()
        curCol[curCol==-1] = fillVals[curCol==-1]
        fillVals[curCol!=-1] = curCol[curCol!=-1]
        final_labels_filled1[:,col] = curCol

    final_labels = np.fmin(final_labels_filled, final_labels_filled1)

    final_labels_smoothed = weightedMedianMatlab(Il, final_labels.copy(), r_median)
    final_labels[occPix==1] = final_labels_smoothed[occPix==1]


    return final_labels
def weightedMedianMatlab(left_img, disp_img, winsize):
    h, w, c = left_img.shape

    smoothed_left_img = np.zeros((h,w,c))
    smoothed_left_img[:,:,0] = medfilt(left_img[:,:,0],3)
    smoothed_left_img[:,:,1] = medfilt(left_img[:,:,1],3)
    smoothed_left_img[:,:,2] = medfilt(left_img[:,:,2],3)

    for d in range(1,max_disp+1):
        # Right to left
        tmp = np.ones((h,w,ch))*threshBorder
        tmp[:,d:w,:] = Ir[:,:w-d,:]
        p_color = abs(tmp - Il)
        p_color = np.mean(p_color, axis = 2)
        p_color = np.minimum(p_color, thresColor)

        tmp = np.ones((h,w))*threshBorder
        tmp[:,d:w] = fx_r[:,:w-d]
        p_grad = abs(tmp - fx_l)
        p_grad = np.minimum(p_grad, thresGrad)

        p = gamma*p_color+(1-gamma)*p_grad
        radius = math.floor(winsize/2.0)

    medianFiltered = np.zeros((h,w))

    for y in range(h):
        for x in range(w):
            maskVals = np.double(filtermask(smoothed_left_img, x, y, winsize))
            dispVals = disp_img[max(0,y-radius):min(h,y+radius),max(0,x-radius):min(w,x+radius)].copy()
            maxDispVal = int(np.amax(dispVals))

            dispVals_f = (dispVals.copy() - 1).astype(np.int).flatten()
            maskVals_f = (maskVals.flatten()).astype(np.double)
            zeros_f = np.zeros(dispVals.shape).astype(np.int).flatten()

            hist = coo_matrix((maskVals_f,(zeros_f,dispVals_f)), shape = (1,maxDispVal)).toarray()
            hist_sum = np.sum(hist)
            hist_cumsum = np.cumsum(hist)

            possbileDispVals = np.arange(1,maxDispVal+1)
            medianval = possbileDispVals[hist_cumsum>(hist_sum/2.0)]
            medianFiltered[y,x] = medianval[0]

    return medianFiltered       

def filtermask(colimg, x, y, winsize):
    radius = math.floor(winsize/2.0)
    h, w, c = colimg.shape
    gamma_c = 0.1
    gamma_d = 9
    patch_h = len( np.arange(max(0,y-radius),min(h,y+radius)) ) 
    patch_w = len( np.arange(max(0,x-radius),min(w,x+radius)) )

    centercol = colimg[y,x,:]
    centerVol = np.zeros((patch_h,patch_w,3))
    centerVol[:,:,0] = centercol[0]
    centerVol[:,:,1] = centercol[1]
    centerVol[:,:,2] = centercol[2]

    Yinds = np.arange(max(0,y-radius),min(h,y+radius), dtype=int)
    patchYinds = np.matlib.repmat( Yinds.reshape(-1,1), 1, patch_w)
    Xinds = np.arange(max(0,x-radius),min(w,x+radius), dtype=int)
    patchXinds = np.matlib.repmat( Xinds, patch_h, 1)

    curPatch = colimg[Yinds[0]:Yinds[-1]+1,Xinds[0]:Xinds[-1]+1,:]
    coldiff = np.sqrt(np.sum( np.square(centerVol - curPatch), axis = 2))

    x_patch = np.ones((patch_h,patch_w))*x
    y_patch = np.ones((patch_h,patch_w))*y
    sdiff = np.sqrt( np.square(x_patch - patchXinds) + np.square(y_patch - patchYinds))

    weights = np.exp(-1* coldiff/(gamma_c*gamma_c)) * np.exp(-1* sdiff/(gamma_d*gamma_d))

    return weights

Y = np.matlib.repmat(np.array(range(h)).reshape(-1,1), 1, w)
X = np.matlib.repmat(np.array(range(w)), h,1 ) - dspLeft
X[X<0] = 0
labelstmp = np.zeros((h,w))
for i in range(h):
    for j in range(w):
        labelstmp[i,j] = dspRight[i,X[i,j]]
final_labels = dspLeft
final_labels[abs(dspLeft - labelstmp)>=1] = -1

inputLabels = final_labels.copy()
final_labels = fillPixelaReference(Il, inputLabels, max_disp) 
# final = weightedMedianMatlab(rgbLeft, dspLeft.copy(), 19)
cv2.imwrite('./depth map/depthMap-final.png', normalize(final_labels))