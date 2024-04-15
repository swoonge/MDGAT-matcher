import os
import numpy as np
import open3d as o3d
import random

sampling_point_num = 16384
voxel_size = -1

for sequence in range(11):
    pc_file = os.path.join('/media/vision/Seagate/DataSets/kitti/dataset/sequences/%02d' % sequence, "velodyne")
    save_path = os.path.join('/media/vision/Seagate/DataSets/kitti/dataset/sequences/%02d' % sequence, "velodyne_randomdownsampled_16384")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for idx, file_name in enumerate(os.listdir(pc_file)):
        if file_name.endswith(".bin"):
            file_path = os.path.join(pc_file, file_name)
            pc = np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))
            
            if voxel_size > 0:
                pc = np.hstack((pc, np.zeros((pc.shape[0], 2))))
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc[:, :3])  # x, y, z
                pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:])  # i

                down_pcd = pcd.voxel_down_sample(voxel_size)
                
                down_np = np.asarray(down_pcd.points)
                colors_np = np.asarray(down_pcd.colors)
                pc = np.hstack((down_np, colors_np))

            pc = pc[np.random.choice(pc.shape[0], sampling_point_num, replace=False), :]

            # Save the downsampled point cloud as a binary file
            save_file_path = os.path.join(save_path, file_name)
            pc.tofile(save_file_path)
            
            if idx % 100 == 0:
                print("Sequence %02d, %d-th frame is done" % (sequence, idx))
    print("Sequence %02d is done." % (sequence))

