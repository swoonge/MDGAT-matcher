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
import copy, pickle

def load_kitti_gt_txt(txt_root, seq):
    '''
    :param txt_root:
    :param seq
    :return: [{anc_idx: *, pos_idx: *, seq: *}]                
     '''
    dataset = []

    with open(os.path.join(txt_root, '%02d'%seq, 'groundtruths.txt'), 'r') as f:
        lines_list = f.readlines()
        for i, line_str in enumerate(lines_list):
            if i == 0:
                # skip the header line
                continue
            line_splitted = line_str.split()
            anc_idx = int(line_splitted[0])
            pos_idx = int(line_splitted[1])

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

        self.gt_seq = opt.seq_num
        self.data_folder = opt.data_folder

        self.dir_SLAM_path = opt.slam_dir + self.data_folder +"/"

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
                        pc = np.fromfile(file, dtype=np.float32)
                        pcs.append(pc)
                    else:
                        pcs.append([0])
                self.pc[sequence] = pcs

    def __len__(self):
        return len(self.poses)
    
    def set_current_pose_idx(self, idx):
        if idx >= len(self.poses):
            idx = len(self.poses) - 1
        self.local_graph_range[1] = idx

        kp_num = 0
        for pc_idx in range(self.local_graph_range[1], 0, -1):
            kp_num += len(self.keypoints[pc_idx])
            if kp_num > 150:
                self.local_graph_range[0] = pc_idx
                break

    def __getitem__(self, idx):
        self.set_current_pose_idx(idx)

        kp0 = []
        kp1 = []
        pc0 = o3d.geometry.PointCloud()
        pc1 = o3d.geometry.PointCloud()
        
        for pc_idx in range(self.local_graph_range[1]):
            pc_chach = o3d.geometry.PointCloud()
            pc_chach.points = o3d.utility.Vector3dVector(np.array(self.dense_scans[pc_idx]))

            if pc_idx >= self.local_graph_range[0]:
                kp1 += self.keypoints[pc_idx]
                pc1 += pc_chach
            else: 
                kp0 += self.keypoints[pc_idx]
                pc0 += pc_chach
        
        pc0 = pc0.voxel_down_sample(voxel_size=0.2)
        pc1 = pc1.voxel_down_sample(voxel_size=0.2)
        pc0 = np.array(pc0.points)
        pc1 = np.array(pc1.points)

        pc0 = pc0.reshape((-1, 8))
        pc1 = pc1.reshape((-1, 8))

        pc0, pc1 = torch.tensor(pc0, dtype=torch.double), torch.tensor(pc1, dtype=torch.double)

        kp0_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in kp0]) 
        kp1_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in kp1])

        kp0_np = torch.tensor(kp0_np, dtype=torch.double)
        kp1_np = torch.tensor(kp1_np, dtype=torch.double)

        kp0_np = kp0_np[:, :3]
        kp1_np = kp1_np[:, :3]        

        return{
            # 'skip': False,
            'keypoints0': kp0_np,
            'keypoints1': kp1_np,
            # 'descriptors0': descs1,
            # 'descriptors1': descs2,
            # 'scores0': scores1_np,
            # 'scores1': scores2_np,
            # 'gt_matches0': match1,
            # 'gt_matches1': match2,
            # 'sequence': sequence,
            # 'idx0': index_in_seq,
            # 'idx1': index_in_seq2,
            # 'pose1': pose1,
            # 'pose2': pose2,
            # 'T_cam0_velo': T_cam0_velo,
            # 'T_gt': T_gt,
            'cloud0': pc0,
            'cloud1': pc1,
            # 'all_matches': list(all_matches),
            # 'file_name': file_name
            # 'rep': rep
        } 


