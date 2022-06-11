import numpy as np
import cv2
from scipy.signal import medfilt
from scipy.sparse import coo_matrix
import math
import numpy.matlib

DEBUG = False

def normalize(mat, minv = 0, maxv = 255):
    min = mat.min()
    max = mat.max()
    mat = np.uint8(((mat - min) / (max - min) * (maxv - minv)) + minv)
    return mat.copy()

def rgb2gray(img):
    return img[:, :, 0] * 0.114 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.2989
# load images
rgbLeft = cv2.imread('./img/tsukuba_rgb_l.png')
rgbRight = cv2.imread('./img/tsukuba_rgb_r.png')
rgbLeft = np.array(rgbLeft, dtype=np.int16)
rgbRight = np.array(rgbRight, dtype=np.int16)
h, w, ch = rgbLeft.shape

######## Cost Volume Filtering ########
print('* Cost Volume Filtering')
r = 9
eps = 0.0001
thresColor = 7/255
thresGrad = 2/255
gamma = 0.11
gamma_c = 0.1
gamma_d = 9
r_median = 11

max_disp = 16

Il_g = rgb2gray(rgbLeft)/255.0
Ir_g = rgb2gray(rgbRight)/255.0
Il = rgbLeft/255.0
Ir = rgbRight/255.0
fx_l = np.gradient(Il_g, axis=1) + 0.5
fx_r = np.gradient(Ir_g, axis=1) + 0.5

def costVolume(rgbLeft, rgbRight, gradLeft, gradRight, max_disp):
    vol = np.ones((max_disp, h,w))
    for k in range(1, max_disp+1):
        tmp = np.ones((h,w,ch))
        tmp[:,k:w,:] = rgbRight[:,:w-k,:] # move right image with disparity k
        dist = abs(tmp - rgbLeft)
        dist = np.mean(dist, axis=2)
        dist = np.minimum(dist, thresColor)

        tmp = np.ones((h,w))
        tmp[:,k:w] = gradRight[:,:w-k]
        distGrad = abs(tmp - gradLeft)
        distGrad = np.minimum(distGrad, thresGrad)

        p = gamma*dist+(1-gamma)*distGrad
        vol[k-1,:,:] = p
        if DEBUG:
            cv2.imwrite('./depth map/filvol%2d.png'% (k-1), normalize(vol[k-1,:,:]))

    for d in range(max_disp):
        p = vol[d,:,:]
        q = cv2.ximgproc.guidedFilter(normalize(rgbLeft), normalize(p), r, eps)
        q = cv2.medianBlur(q, 5).copy()
        vol[d,:,:] = q
    dis = (np.argmin(vol, axis = 0) + np.ones((h,w)) ).astype(int)
    return dis.copy()
dspLeft = costVolume(Il, Ir, fx_l, fx_r, max_disp)
dspRight = np.fliplr(costVolume(np.fliplr(Ir), np.fliplr(Il), np.fliplr(fx_r), np.fliplr(fx_l), max_disp))
if DEBUG:
    cv2.imwrite('./depth map/depthMap_l.png', normalize(dspLeft))
    cv2.imwrite('./depth map/depthMap_r.png', normalize(dspRight))

######## Disparity Consistancy ########
print('* Disparity Consistancy')
check = np.zeros((h, w))
for i in range(rgbLeft.shape[0]):
    for j in range(rgbLeft.shape[1]):
        if j - dspLeft[i][j] >= rgbLeft.shape[1] or j - dspLeft[i][j] < 0:
            continue
        if abs(dspLeft[i][j] - dspRight[i][j - dspLeft[i][j]]) < 2:
            check[i][j] = 255
check = np.uint8(check)
if DEBUG:
    cv2.imwrite('./depth map/depthMap-check.png', check)

####### Hole Filling ########
print('* Hole Filling')
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
# cv2.imwrite('./depth map/depthMap-fill.png', normalize(after_fill))
after_fill = cv2.medianBlur(normalize(after_fill), 7)
if DEBUG:
    cv2.imwrite('./depth map/depthMap-fill2.png', after_fill)

####### Weighted Medium Filtering ########
print("* Weighted Medium Filtering")

def weightedMedianMatlab(left_img, disp_img, winsize):
    h, w, c = left_img.shape

    smoothed_left_img = np.zeros((h,w,c))
    smoothed_left_img[:,:,0] = medfilt(left_img[:,:,0],3)
    smoothed_left_img[:,:,1] = medfilt(left_img[:,:,1],3)
    smoothed_left_img[:,:,2] = medfilt(left_img[:,:,2],3)

    for d in range(1,max_disp+1):
        # Right to left
        tmp = np.ones((h,w,ch))
        tmp[:,d:w,:] = Ir[:,:w-d,:]
        p_color = abs(tmp - Il)
        p_color = np.mean(p_color, axis = 2)
        p_color = np.minimum(p_color, thresColor)

        tmp = np.ones((h,w))
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

            dispVals_f = (dispVals.copy() - 1).astype(np.int16).flatten()
            maskVals_f = (maskVals.flatten()).astype(np.double)
            zeros_f = np.zeros(dispVals.shape).astype(np.int16).flatten()

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

after_fill = normalize(after_fill, 1, 16)
final = weightedMedianMatlab(Il, after_fill, 25)
cv2.imwrite('./depth map/depthMap.png', normalize(final))