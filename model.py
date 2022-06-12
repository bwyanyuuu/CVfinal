import cv2
import numpy as np
import open3d as o3d
#import trimesh

def save_ply(Z,color,background,filepath):
    Z_map = np.reshape(Z, (Z.shape[0],Z.shape[1])).copy()
    data = np.zeros((Z.shape[0]*Z.shape[1]*2,3),dtype=np.float32)
    img_color = np.zeros((Z.shape[0]*Z.shape[1]*2,3),dtype=np.float32)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    #Z_map[np.where(Z_map != 0)] *= 2
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            idx = i * Z.shape[1] + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[Z.shape[0] - 1 - i][j]/5
            img_color[idx][0] = color[Z.shape[0] - 1 - i][j][2]/255
            img_color[idx][1] = color[Z.shape[0] - 1 - i][j][1]/255
            img_color[idx][2] = color[Z.shape[0] - 1 - i][j][0]/255
            data[idx+Z.shape[0]*Z.shape[1]][0] = j
            data[idx+Z.shape[0]*Z.shape[1]][1] = i
            data[idx+Z.shape[0]*Z.shape[1]][2] = (baseline_val+75)/5
            img_color[idx+Z.shape[0]*Z.shape[1]][0] = background[Z.shape[0] - 1 - i][j][2]/255
            img_color[idx+Z.shape[0]*Z.shape[1]][1] = background[Z.shape[0] - 1 - i][j][1]/255
            img_color[idx+Z.shape[0]*Z.shape[1]][2] = background[Z.shape[0] - 1 - i][j][0]/255
    # st = Z.shape[0]*Z.shape[1]
    # for i in range(Z.shape[0]*2):
    #     for j in range(Z.shape[1]*2):
    #         idx = i * Z.shape[1]*2 + j
    #         data[st+idx][0] = j-int(Z.shape[1]/2)
    #         data[st+idx][1] = i-int(Z.shape[0]/2)
    #         data[st+idx][2] = baseline_val+55
    #         img_color[st+idx][0] = color[5][320][2]/255
    #         img_color[st+idx][1] = color[5][320][1]/255
    #         img_color[st+idx][2] = color[5][320][0]/255

    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    pcd.colors = o3d.utility.Vector3dVector(img_color)
    o3d.io.write_point_cloud(filepath, pcd,write_ascii=True)

    ########################################################

    # pcd.estimate_normals()

    # # estimate radius for rolling ball
    # distances = pcd.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radius = 1.5 * avg_dist   

    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #            pcd,
    #            o3d.utility.DoubleVector([radius, radius * 2]))
    # mesh.triangle_uvs = 

    # # create the triangular mesh with the vertices and faces from open3d
    # tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
    #                           vertex_normals=np.asarray(mesh.vertex_normals))

    # trimesh.convex.is_convex(tri_mesh)
    # trimesh.exchange.export.export_mesh(tri_mesh, './model/trimesh2.obj')

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])
    # alpha = 0.002
    # print(f"alpha={alpha:.3f}")
    # tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    # o3d.visualization.draw_geometries([pcd,tetra_mesh], mesh_show_back_face=True)
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha, tetra_mesh, pt_map)
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


img = cv2.imread('./depth map/depthMap-final.png', 0)
color = cv2.imread('./img/tsukuba_rgb_l.png')

background = cv2.imread('./output/f1_output.png')
print(color.shape)


save_ply(img, color, background,  './model/fin2.ply')
show_ply('./model/fin2.ply')

