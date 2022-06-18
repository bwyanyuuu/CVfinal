import numpy as np
import cv2

img = cv2.imread('./img/tsukuba_rgb_l.png')
depthmap = cv2.imread('./depth map/depthMap-final.png',0)

# img = cv2.imread('./img/view6.png')
# depthmap = cv2.imread('./depth map/bolling/depthMap-final.png',0)

print(img.shape, depthmap.shape)
print(np.max(img), np.max(depthmap))

cutlist = {}

similar = 15
batch = 35 #number of output images
e = 3


for i in range(depthmap.shape[0]):
	for j in range(depthmap.shape[1]):
		depthmap[i][j] = int(depthmap[i][j]/similar)*similar


for i in range(depthmap.shape[0]):
	for j in range(depthmap.shape[1]):
		if depthmap[i][j] not in cutlist.keys():
			cutlist[depthmap[i][j]] = 0
		cutlist[depthmap[i][j]] += 1

sortlist = sorted(cutlist.items(), key=lambda x:x[1])



if len(sortlist)<batch:
	batch = len(sortlist)
maxlist = []
for i in range(batch):
	maxlist.append(sortlist[-i][0])  

print(maxlist)
for m in reversed(range(batch)):
	for i in range(depthmap.shape[0]):
		for j in range(depthmap.shape[1]):
			if ((depthmap[i][j]<= maxlist[m]+e) and (depthmap[i][j]>= maxlist[m]-e)):
				depthmap[i][j] = maxlist[m]


for m in range(len(maxlist)):
	mask = depthmap == maxlist[m]
	if len(img.shape)>2:
		mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
	output = img*mask
	cv2.imwrite('./output/f'+str(m)+'.png', output)

fp = open("./output/value.txt", "w")
for i in maxlist:
	fp.write(str(i)+'\n')
fp.close()

