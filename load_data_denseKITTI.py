import numpy as np
import torch
import os
import open3d as o3d
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset
import open3d as o3d 
from sklearn.neighbors import KDTree
import time
import random
import math

def generate_random_rotation():
    theta = torch.rand(1).item() * 2 * math.pi
    random_rot = torch.tensor([
        [math.cos(theta), -math.sin(theta), 0, 0],
        [math.sin(theta), math.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.double)
    return random_rot

def load_kitti_gt_txt(txt_root, seq):
    '''
    :param txt_root:
    :param seq
    :return: [{anc_idx: *, pos_idx: *, seq: *}]                
     '''
    dataset = []
    file_path = os.path.join(txt_root, 'groundtruths128_v2', '%02d'%seq, 'groundtruths.txt')
    with open(file_path, 'r') as f:
        lines_list = f.readlines()
        for i, line_str in enumerate(lines_list):
            if i == 0:
                # skip the header line
                continue
            line_splitted = line_str.split()
            anc_idx = int(float(line_splitted[0]))
            pos_idx = int(float(line_splitted[1]))

            data = {'seq': seq, 'anc_idx': anc_idx, 'pos_idx': pos_idx}
            dataset.append(data)
    # dataset.pop(0)
    return dataset

def make_dataset_kitti_distance(txt_path, mode):
        if mode == 'train':
            seq_list = list([0,2,3,4,5,6,7])
        elif mode == 'val':
            seq_list = [9]
        elif mode == 'test':
            seq_list = [10]
        else:
            raise Exception('Invalid mode.')

        dataset = []
        for seq in seq_list:
            dataset += (load_kitti_gt_txt(txt_path, seq))
           
        return dataset, seq_list

class SparseDataset(Dataset):
    """Sparse correspondences dataset.  
    Reads images from files and creates pairs. It generates keypoints, 
    descriptors and ground truth matches which will be used in training."""

    def __init__(self, opt, mode):
        self.flag_count = 0
        self.train_path = opt.train_path
        self.keypoints = opt.keypoints
        self.keypoints_path = opt.keypoints_path
        self.descriptor = opt.descriptor
        self.nfeatures = opt.max_keypoints
        self.threshold = opt.threshold
        self.ensure_kpts_num = opt.ensure_kpts_num
        self.mutual_check = opt.mutual_check
        self.memory_is_enough = opt.memory_is_enough
        self.txt_path = opt.txt_path
        self.dataset, self.seq_list = make_dataset_kitti_distance(self.txt_path, mode)

        self.calib={}
        self.pose={}
        self.pc = {}
        self.random_sample_num = 16384
        
        for seq in self.seq_list:
            sequence = '%02d'%seq
            calibpath = os.path.join(self.train_path, 'calib/sequences', sequence, 'calib.txt')
            posepath = os.path.join(self.train_path, 'poses', '%02d.txt'%seq)
            with open(calibpath, 'r') as f:
                for line in f.readlines():
                    _, value = line.split(':', 1)
                    try:
                        calib = np.array([float(x) for x in value.split()])
                    except ValueError:
                        pass
                    calib = np.reshape(calib, (3, 4))    
                    self.calib[sequence] = np.vstack([calib, [0, 0, 0, 1]])
            
            poses = []
            with open(posepath, 'r') as f:
                for line in f.readlines():
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)
                self.pose[sequence] = poses

            '''If memory is enough, load all the data'''
            if self.memory_is_enough:
                pcs = []
                folder = os.path.join(self.keypoints_path, sequence)
                folder = os.listdir(folder)   
                folder.sort(key=lambda x:int(x[:-4]))
                for idx in range(len(folder)):
                    file = os.path.join(self.keypoints_path, sequence, folder[idx])
                    if os.path.isfile(file):
                        # pc = np.reshape(np.fromfile(file, dtype=np.float64), (-1, 139))
                        pc = np.fromfile(file, dtype=np.float64)
                        # print(pc.shape)
                        pcs.append(pc)
                    else:
                        pcs.append([0])
                self.pc[sequence] = pcs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        index_in_seq0 = self.dataset[idx]['anc_idx']
        index_in_seq1 = self.dataset[idx]['pos_idx']
       
        seq = self.dataset[idx]['seq']
        sequence = '%02d'%seq

        pc_file0 = os.path.join('/media/vision/Seagate/DataSets/denseKITTI/dense_scan', sequence, '%06d.bin' % index_in_seq0)
        pc_file1 = os.path.join('/media/vision/Seagate/DataSets/denseKITTI/dense_scan', sequence, '%06d.bin' % index_in_seq1)
        pc0_w = np.reshape(np.fromfile(pc_file0, dtype=np.float64), (-1, 3))
        pc1_w = np.reshape(np.fromfile(pc_file1, dtype=np.float64), (-1, 3))
        pc0_w = pc0_w[np.random.choice(pc0_w.shape[0], 20000, replace=False), :] # 16384
        pc1_w = pc1_w[np.random.choice(pc1_w.shape[0], 20000, replace=False), :]
        pc0_w = np.array([(kp[0], kp[1], kp[2], 1) for kp in pc0_w]) 
        pc1_w = np.array([(kp[0], kp[1], kp[2], 1) for kp in pc1_w]) 
        pc0_w, pc1_w = torch.tensor(pc0_w, dtype=torch.double), torch.tensor(pc1_w, dtype=torch.double)

        # relative_pos = self.dataset[idx]['anc_idx']
        # repeat until the number of keypoints is enough ( > 20)
        while True:
            sequence = '%02d'%seq
            pc_np0 = self.pc[sequence][index_in_seq0]
            pc_np0 = pc_np0.reshape((-1, 139))
            random_sample_indices0 = np.random.choice(len(pc_np0), 128, replace=False)
            
            kp0_w = pc_np0[random_sample_indices0, :3]
            score0 = pc_np0[random_sample_indices0, 3]
            descs0 = pc_np0[random_sample_indices0, 4:]
            pose0 = self.pose[sequence][index_in_seq0] 

            pc_np1 = self.pc[sequence][index_in_seq1]
            pc_np1 = pc_np1.reshape((-1, 139))
            random_sample_indices1 = np.random.choice(len(pc_np1), 128, replace=False)

            kp1_w = pc_np1[random_sample_indices1, :3]
            score1 = pc_np1[random_sample_indices1, 3]
            descs1 = pc_np1[random_sample_indices1, 4:]
            pose1 = self.pose[sequence][index_in_seq1]

            T_cam0_velo = self.calib[sequence]
            # q = np.asarray([rot[3], rot[0], rot[1], rot[2]])
            # t = np.asarray(trans)
            # relative_pose = RigidTransform(q, t)

            kp0_num = len(kp0_w)
            kp1_num = len(kp1_w)
            kp0_w_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in kp0_w])
            kp1_w_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in kp1_w])
            
            scores0_np = np.array(score0) 
            scores1_np = np.array(score1)

            kp0_w_np = torch.tensor(kp0_w_np, dtype=torch.double)
            pose0 = torch.tensor(pose0, dtype=torch.double)
            kp1_w_np = torch.tensor(kp1_w_np, dtype=torch.double)
            pose1 = torch.tensor(pose1, dtype=torch.double)
            T_cam0_velo = torch.tensor(T_cam0_velo, dtype=torch.double)
            
            '''transform pose from cam0 to LiDAR'''
            # add random rotation to the point cloud and keypoints (z-axis rotation only)
            pose0 = torch.einsum('ij,jk->ik', generate_random_rotation(), pose0)
            pose1 = torch.einsum('ij,jk->ik', generate_random_rotation(), pose1)

            kp0_np = torch.einsum('ij,nj->ni', torch.inverse(pose0), kp0_w_np)
            kp1_np = torch.einsum('ij,nj->ni', torch.inverse(pose1), kp1_w_np)
            pc0 = torch.einsum('ij,nj->ni', torch.inverse(pose0), pc0_w)
            pc1 = torch.einsum('ij,nj->ni', torch.inverse(pose1), pc1_w)
            T_gt = torch.einsum('ab,de->ae', torch.inverse(pose0), pose1) # T_gt: transpose kp2 to kp1
            
            kp0_w_np = kp0_w_np[:, :3]
            kp1_w_np = kp1_w_np[:, :3]
            kp0_np = kp0_np[:, :3]
            kp1_np = kp1_np[:, :3]
            pc0_w = pc0_w[:, :3]
            pc1_w = pc1_w[:, :3]
            pc0 = pc0[:, :3]
            pc1 = pc1[:, :3]

            vis_registered_keypoints = False
            if vis_registered_keypoints:
                point_cloud_o3d = o3d.geometry.PointCloud()
                point_cloud_o3d.points = o3d.utility.Vector3dVector(kp0_w_np.numpy())
                point_cloud_o3d.paint_uniform_color([0, 1, 0])
                point_cloud_o3d2 = o3d.geometry.PointCloud()
                point_cloud_o3d2.points = o3d.utility.Vector3dVector(kp1_w_np.numpy())
                point_cloud_o3d2.paint_uniform_color([1, 0, 0])
                o3d.visualization.draw_geometries([point_cloud_o3d, point_cloud_o3d2])

            dists = cdist(kp0_w_np, kp1_w_np)

            '''Find ground true keypoint matching'''
            min0 = np.argmin(dists, axis=0)
            min1 = np.argmin(dists, axis=1)
            min0v = np.min(dists, axis=1)
            min0f = min1[min0v < self.threshold]

            '''For calculating repeatibility'''
            rep = len(min0f)

            if rep > 20:
                break
            else:
                pc0_w = torch.cat((pc0_w, torch.ones(pc0_w.shape[0], 1)), dim=1)
                pc1_w = torch.cat((pc1_w, torch.ones(pc1_w.shape[0], 1)), dim=1)
                pass

        '''
        If you got high-quality keypoints, you can set the 
        mutual_check to True, otherwise, it is better to 
        set to False
        '''
        match0, match1 = -1 * np.ones((len(kp0_w_np)), dtype=np.int16), -1 * np.ones((len(kp1_w_np)), dtype=np.int16)
        if self.mutual_check:
            xx = np.where(min1[min0] == np.arange(min0.shape[0]))[0]
            matches = np.intersect1d(min0f, xx)

            match0[min0[matches]] = matches
            match1[matches] = min0[matches]
        else:
            match1[min0v < self.threshold] = min0f

            min1v = np.min(dists, axis=0)
            min1f = min0[min1v < self.threshold]
            match1[min1v < self.threshold] = min1f

        kp0_np = kp0_np[:, :3]
        kp1_np = kp1_np[:, :3]

        # descs1, descs2  = np.multiply(descs1, 1/norm1), np.multiply(descs2, 1/norm2)
        norm0, norm1 = np.linalg.norm(descs0, axis=1), np.linalg.norm(descs1, axis=1)
        norm0, norm1 = norm0.reshape(kp0_num, 1), norm1.reshape(kp1_num, 1)
        epsilon = 1e-8  # small constant to prevent division by zero
        norm0, norm1 = norm0 + epsilon, norm1 + epsilon
        descs0, descs1 = np.where(norm0 != 0, np.multiply(descs0, 1/norm0), 0), np.where(norm1 != 0, np.multiply(descs1, 1/norm1), 0)
        
        descs0, descs1 = torch.tensor(descs0, dtype=torch.double), torch.tensor(descs1, dtype=torch.double)
        scores0_np, scores1_np = torch.tensor(scores0_np, dtype=torch.double), torch.tensor(scores1_np, dtype=torch.double)

        return{
            # 'skip': False,
            'keypoints0': kp0_np,
            'keypoints1': kp1_np,
            'descriptors0': descs0,
            'descriptors1': descs1,
            'scores0': scores0_np,
            'scores1': scores1_np,
            'gt_matches0': match0,
            'gt_matches1': match1,
            'sequence': sequence,
            'idx0': index_in_seq0,
            'idx1': index_in_seq1,
            'pose1': pose0,
            'pose2': pose1,
            'T_cam0_velo': T_cam0_velo,
            'T_gt': T_gt,
            'cloud0': pc0,
            'cloud1': pc1,
            # 'all_matches': list(all_matches),
            # 'file_name': file_name
            'rep': rep
        } 


