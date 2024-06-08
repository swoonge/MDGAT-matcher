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
        # d=[9,236,390,259,1048,171,395,296]
        # d=[259,296]
        # idx=d[idx]
        index_in_seq = self.dataset[idx]['anc_idx']
        index_in_seq2 = self.dataset[idx]['pos_idx']
       
        seq = self.dataset[idx]['seq']
        sequence = '%02d'%seq
        # trans = self.dataset[idx]['trans']
        # rot = self.dataset[idx]['rot']
        # print("seq: ", seq, "idx: ", index_in_seq, "idx2: ", index_in_seq2)
        pc_file1 = os.path.join('/media/vision/Seagate/DataSets/denseKITTI/dense_scan', sequence, '%06d.bin' % index_in_seq)
        pc_file2 = os.path.join('/media/vision/Seagate/DataSets/denseKITTI/dense_scan', sequence, '%06d.bin' % index_in_seq2)
        
        pc1_w = np.reshape(np.fromfile(pc_file1, dtype=np.float64), (-1, 3))
        pc2_w = np.reshape(np.fromfile(pc_file2, dtype=np.float64), (-1, 3))

        pc1_w = pc1_w[np.random.choice(pc1_w.shape[0], 20000, replace=False), :] # 16384
        pc2_w = pc2_w[np.random.choice(pc2_w.shape[0], 20000, replace=False), :]

        pc1_w = np.array([(kp[0], kp[1], kp[2], 1) for kp in pc1_w]) 
        pc2_w = np.array([(kp[0], kp[1], kp[2], 1) for kp in pc2_w]) 

        pc1_w, pc2_w = torch.tensor(pc1_w, dtype=torch.double), torch.tensor(pc2_w, dtype=torch.double)

        # relative_pos = self.dataset[idx]['anc_idx']
        while True:
            if self.memory_is_enough: # If memory is enough, load all the data -> True
                sequence = '%02d'%seq
                pc_np1 = self.pc[sequence][index_in_seq]
                pc_np1 = pc_np1.reshape((-1, 139))
                random_sample_indices1 = np.random.choice(len(pc_np1), 128, replace=False)
                
                kp1_w = pc_np1[random_sample_indices1, :3]
                score1 = pc_np1[random_sample_indices1, 3]
                descs1 = pc_np1[random_sample_indices1, 4:]
                pose1 = self.pose[sequence][index_in_seq] 

                pc_np2 = self.pc[sequence][index_in_seq2]
                pc_np2 = pc_np2.reshape((-1, 139))
                random_sample_indices2 = np.random.choice(len(pc_np2), 128, replace=False)

                kp2_w = pc_np2[random_sample_indices2, :3]
                score2 = pc_np2[random_sample_indices2, 3]
                descs2 = pc_np2[random_sample_indices2, 4:]
                pose2 = self.pose[sequence][index_in_seq2]

                T_cam0_velo = self.calib[sequence]
                # q = np.asarray([rot[3], rot[0], rot[1], rot[2]])
                # t = np.asarray(trans)
                # relative_pose = RigidTransform(q, t)
            else:
                sequence = '%02d'%seq
                pc_np_file1 = os.path.join(self.keypoints_path, sequence, '%06d.bin' % (index_in_seq))
                pc_np1 = np.fromfile(pc_np_file1, dtype=np.float32)

                pc_np_file2 = os.path.join(self.keypoints_path, sequence, '%06d.bin' % (index_in_seq2))
                pc_np2 = np.fromfile(pc_np_file2, dtype=np.float32)
                
                pc_np1 = pc_np1.reshape((-1, 139))
                kp1_w = pc_np1[:, :3]

                pc_np2 = pc_np2.reshape((-1, 139))
                kp2_w = pc_np2[:, :3]

                score1 = pc_np1[:, 3]
                descs1 = pc_np1[:, 4:]
                # pose1 = dataset.poses[index_in_seq]
                pose1 = self.pose[sequence][index_in_seq]
                # pc1 = dataset.get_velo(index_in_seq)

                score2 = pc_np2[:, 3]
                descs2 = pc_np2[:, 4:]
                # pose2 = dataset.poses[index_in_seq2]
                pose2 = self.pose[sequence][index_in_seq2]

                T_cam0_velo = self.calib[sequence]

            kp1_num = len(kp1_w)
            kp2_num = len(kp2_w)
            kp1_w_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in kp1_w]) 
            kp2_w_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in kp2_w])
            
            scores1_np = np.array(score1) 
            scores2_np = np.array(score2)

            kp1_w_np = torch.tensor(kp1_w_np, dtype=torch.double)
            # kp1_w_np = torch.cat([kp1_w_np, torch.ones((kp1_w_np.shape[0], 1), dtype=kp1_w_np.dtype)], dim=1)
            pose1 = torch.tensor(pose1, dtype=torch.double)
            kp2_w_np = torch.tensor(kp2_w_np, dtype=torch.double)
            # kp2_w_np = torch.cat([kp2_w_np, torch.ones((kp2_w_np.shape[0], 1), dtype=kp2_w_np.dtype)], dim=1)
            pose2 = torch.tensor(pose2, dtype=torch.double)
            T_cam0_velo = torch.tensor(T_cam0_velo, dtype=torch.double)
            T_gt = torch.einsum('ab,de->ae', torch.inverse(pose1), pose2) # T_gt: transpose kp2 to kp1

            '''transform pose from cam0 to LiDAR'''
            # kp1_np = torch.einsum('ki,ij,jm->mk', pose1, T_cam0_velo, kp1_w_np.T)
            # kp2_np = torch.einsum('ki,ij,jm->mk', pose2, T_cam0_velo, kp2_w_np.T)
            kp1_np = torch.einsum('ij,nj->ni', torch.inverse(pose1), kp1_w_np)
            kp2_np = torch.einsum('ij,nj->ni', torch.inverse(pose2), kp2_w_np)
            pc1 = torch.einsum('ij,nj->ni', torch.inverse(pose1), pc1_w)
            pc2 = torch.einsum('ij,nj->ni', torch.inverse(pose2), pc2_w)
            
            kp1_w_np = kp1_w_np[:, :3]
            kp2_w_np = kp2_w_np[:, :3]
            kp1_np = kp1_np[:, :3]
            kp2_np = kp2_np[:, :3]
            pc1 = pc1[:, :3]
            pc2 = pc2[:, :3]

            vis_registered_keypoints = False
            if vis_registered_keypoints:
                point_cloud_o3d = o3d.geometry.PointCloud()
                point_cloud_o3d.points = o3d.utility.Vector3dVector(kp1_w_np.numpy())
                point_cloud_o3d.paint_uniform_color([0, 1, 0])
                point_cloud_o3d2 = o3d.geometry.PointCloud()
                point_cloud_o3d2.points = o3d.utility.Vector3dVector(kp2_w_np.numpy())
                point_cloud_o3d2.paint_uniform_color([1, 0, 0])
                o3d.visualization.draw_geometries([point_cloud_o3d, point_cloud_o3d2])

            dists = cdist(kp1_w_np, kp2_w_np)

            '''Find ground true keypoint matching'''
            min1 = np.argmin(dists, axis=0)
            min2 = np.argmin(dists, axis=1)
            min1v = np.min(dists, axis=1)
            min1f = min2[min1v < self.threshold]

            '''For calculating repeatibility'''
            rep = len(min1f)

            if rep > 20:
                break
            else:
                pass
                # print()
                # print("rep < 10: ", rep)

        '''
        If you got high-quality keypoints, you can set the 
        mutual_check to True, otherwise, it is better to 
        set to False
        '''
        match1, match2 = -1 * np.ones((len(kp1_w_np)), dtype=np.int16), -1 * np.ones((len(kp2_w_np)), dtype=np.int16)
        if self.mutual_check:
            xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
            matches = np.intersect1d(min1f, xx)

            match1[min1[matches]] = matches
            match2[matches] = min1[matches]
        else:
            match1[min1v < self.threshold] = min1f

            min2v = np.min(dists, axis=0)
            min2f = min1[min2v < self.threshold]
            match2[min2v < self.threshold] = min2f

        kp1_np = kp1_np[:, :3]
        kp2_np = kp2_np[:, :3]

        # descs1, descs2  = np.multiply(descs1, 1/norm1), np.multiply(descs2, 1/norm2)
        norm1, norm2 = np.linalg.norm(descs1, axis=1), np.linalg.norm(descs2, axis=1)
        norm1, norm2 = norm1.reshape(kp1_num, 1), norm2.reshape(kp2_num, 1)
        epsilon = 1e-8  # small constant to prevent division by zero
        norm1, norm2 = norm1 + epsilon, norm2 + epsilon
        descs1, descs2 = np.where(norm1 != 0, np.multiply(descs1, 1/norm1), 0), np.where(norm2 != 0, np.multiply(descs2, 1/norm2), 0)
        # kp1_max_dist = torch.max(torch.norm(kp1_np[:, :3], dim=1))
        # kp1_np = kp1_np / kp1_max_dist
        # kp2_max_dist = torch.max(torch.norm(kp2_np[:, :3], dim=1))
        # kp2_np = kp2_np / kp2_max_dist
        
        descs1, descs2 = torch.tensor(descs1, dtype=torch.double), torch.tensor(descs2, dtype=torch.double)
        scores1_np, scores2_np = torch.tensor(scores1_np, dtype=torch.double), torch.tensor(scores2_np, dtype=torch.double)

        return{
            # 'skip': False,
            'keypoints0': kp1_np,
            'keypoints1': kp2_np,
            'descriptors0': descs1,
            'descriptors1': descs2,
            'scores0': scores1_np,
            'scores1': scores2_np,
            'gt_matches0': match1,
            'gt_matches1': match2,
            'sequence': sequence,
            'idx0': index_in_seq,
            'idx1': index_in_seq2,
            'pose1': pose1,
            'pose2': pose2,
            'T_cam0_velo': T_cam0_velo,
            'T_gt': T_gt,
            'cloud0': pc1,
            'cloud1': pc2,
            # 'all_matches': list(all_matches),
            # 'file_name': file_name
            'rep': rep
        } 


