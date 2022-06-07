import cv2
import numpy as np
import open3d as o3d

def save_ply(Z,color,filepath):
    Z_map = np.reshape(Z, (Z.shape[0],Z.shape[1])).copy()
    data = np.zeros((Z.shape[0]*Z.shape[1],3),dtype=np.float32)
    img_color = np.zeros((Z.shape[0]*Z.shape[1],3),dtype=np.float32)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    #Z_map[np.where(Z_map != 0)] *= 2
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            idx = i * Z.shape[1] + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[Z.shape[0] - 1 - i][j]
            img_color[idx][0] = color[Z.shape[0] - 1 - i][j][2]/255
            img_color[idx][1] = color[Z.shape[0] - 1 - i][j][1]/255
            img_color[idx][2] = color[Z.shape[0] - 1 - i][j][0]/255
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    pcd.colors = o3d.utility.Vector3dVector(img_color)
    o3d.io.write_point_cloud(filepath, pcd,write_ascii=True)

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])


img = cv2.imread('./depth map/tsukuba.png', 0)
color = cv2.imread('./img/ImL.png')
print(color.shape)


save_ply(img, color,  './model/depthMap2.ply')
show_ply('./model/depthMap2.ply')