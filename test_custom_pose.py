#!/usr/bin/env python3
import os, sys, signal,rospy, argparse, csv

from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
import copy, pickle
import open3d as o3d
import torch

from open3d_ros_helper import open3d_ros_helper as orh
from geometry_msgs.msg import Pose, PoseArray, Point # PoseArray, Pose
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker

from models.mdgat import MDGAT

# parser = argparse.ArgumentParser(description='A simple kitti publisher')
# parser.add_argument('--gt_dir', type=str, default='/media/vision/Seagate/DataSets/kitti/dataset/sequences/', metavar='DIR', help='path to dataset')

torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(
    description='Point cloud matching and pose evaluation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--slam_dir', type=str, default='/media/vision/Seagate/DataSets/KRGM/kitti/', metavar='DIR', help='path to SLAM dataset')
parser.add_argument('--data_folder', type=str, default='harris_3D', metavar='DIR', help='path to SLAM dataset')
parser.add_argument('--local_global', type=bool, default=False, help='')
parser.add_argument('--seq_num', type=str, default='00', help='seq_num')

parser.add_argument(
    '--visualize', type=bool, default=False,
    help='Visualize the matches')

parser.add_argument(
    '--vis_line_width', type=float, default=0.2,
    help='the width of the match line open3d visualization')

parser.add_argument(
    '--calculate_pose', type=bool, default=True,
    help='Registrate the point cloud using the matched point pairs and calculate the pose')

parser.add_argument(
    '--learning_rate', type=int, default=0.0001,
    help='Learning rate')
    
parser.add_argument(
    '--batch_size', type=int, default=1,
    help='batch_size')

parser.add_argument(
    '--train_path', type=str, default='./KITTI/',
    help='Path to the directory of training scans.')

parser.add_argument(
    '--model_out_path', type=str, default='./models/checkpoint',
    help='Path to the directory of output model')

parser.add_argument(
    '--memory_is_enough', type=bool, default=False, 
    help='If true load all the scans')

parser.add_argument(
    '--local_rank', type=int, default=0, 
    help='Gpu rank.')

parser.add_argument(
    '--txt_path', type=str, default='./KITTI/preprocess-random-full',
    help='Path to the directory of kepoints.')

parser.add_argument(
    '--keypoints_path', type=str, default='./KITTI/keypoints/tsf_256_FPFH_16384-512-k1k16-2d-nonoise',
    help='Path to the directory of kepoints.')

parser.add_argument(
    '--resume_model', type=str, default='./checkpoint/kitti/mdgat-l9-gap_loss-pointnetmsg-04_01_19_32/train_step3/nomutualcheck-mdgat-batch16-gap_loss-pointnetmsg-USIP-04_01_19_32/best_model_epoch_221(val_loss0.31414552026539594).pth',
    help='Number of skip frames for training')

parser.add_argument(
    '--loss_method', type=str, default='triplet_loss', 
    help='triplet_loss superglue gap_loss')

parser.add_argument(
    '--net', type=str, default='mdgat', 
    help='mdgat; superglue')

parser.add_argument(
    '--mutual_check', type=bool, default=False,
    help='perform')

parser.add_argument(
    '--k', type=int, default=[128, None, 128, None, 64, None, 64, None], 
    help='Mdgat structure. None means connect all the nodes.')

parser.add_argument(
    '--l', type=int, default=9, 
    help='Layers number in GNN')

parser.add_argument(
    '--descriptor', type=str, default='pointnetmsg', 
    help='FPFH pointnet FPFH_only msg')

parser.add_argument(
    '--keypoints', type=str, default='USIP', 
    help='USIP')

parser.add_argument(
    '--ensure_kpts_num', type=bool, default=False, 
    help='make kepoints number')

parser.add_argument(
    '--max_keypoints', type=int, default=-1,
    help='Maximum number of keypoints'
            ' (\'-1\' keeps all keypoints)')

parser.add_argument(
    '--match_threshold', type=float, default=0.2,
    help='SuperGlue match threshold')

parser.add_argument(
    '--threshold', type=float, default=0.5, 
    help='Ground truth distance threshold')

parser.add_argument(
    '--triplet_loss_gamma', type=float, default=0.5,
    help='Threshold for triplet loss and gap loss')

parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')

parser.add_argument(
    '--train_step', type=int, default=3,  
    help='Training step when using pointnet: 1,2,3')

class dataset():
    def __init__(self, args) -> None:
        self.gt_seq = args.seq_num
        self.data_folder = args.data_folder

        self.dir_SLAM_path = args.slam_dir + self.data_folder +"/"
        self.pub_SLAM_map = rospy.Publisher('/slam_map', PointCloud2, queue_size=100)
        self.pub_SLAM_keypoints = rospy.Publisher('/slam_keypoints', PointCloud2, queue_size=100)
        self.pub_SLAM_keypoints_local = rospy.Publisher('/slam_keypoints_local', PointCloud2, queue_size=100)
        self.pub_SLAM_poses = rospy.Publisher('/slam_odom', PoseArray, queue_size=100)
        # self.pub_matching_line = rospy.Publisher('/matching_line', Marker, queue_size=100)

        self.poses = []
        self.dense_scans = []
        self.keypoints = []
        self.descriptors = []
        self.local_graph_range = [0, 0]

        self._get_SLAM_poses()
        self._get_dense_frames()
        self._get_keypoints()
        # self._get_descriptors()
        print("[Load] SLAM data complite")

    def _get_SLAM_poses(self):
        with open(file=os.path.join(self.dir_SLAM_path, "Poses_kitti_" + self.gt_seq + ".pickle"), mode='rb') as f:
            self.poses = pickle.load(f)
        print("poses: ", len(self.poses))

    def _get_dense_frames(self):
        with open(file=os.path.join(self.dir_SLAM_path, "DenseFrames_kitti_" + self.gt_seq + ".pickle"), mode='rb') as f:
            self.dense_scans = pickle.load(f)

    def _get_keypoints(self):
        with open(file=os.path.join(self.dir_SLAM_path, "keyPoints_kitti_" + self.gt_seq + ".pickle"), mode='rb') as f:
            self.keypoints = pickle.load(f)

    def _get_descriptors(self):
        with open(file=os.path.join(self.dir_SLAM_path, "Descriptors_FPFH_kitti_" + self.gt_seq + ".pickle"), mode='rb') as f:
            self.descriptors = pickle.load(f)

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

    def _keypoints_l2_matching(self):
        local_keypoints = np.zeros((1,3))
        local_descriptors = np.zeros((1,33))
        global_keypoints = np.zeros((1,3))
        global_descriptors = np.zeros((1,33))

        for idx in range(self.local_graph_range[0], self.local_graph_range[1]):
            if self.keypoints[idx].shape[0]: local_keypoints = np.vstack((local_keypoints, self.keypoints[idx]))
            if len(self.descriptors[idx]): local_descriptors = np.vstack((local_descriptors, self.descriptors[idx]))
        for idx in range(self.local_graph_range[0]):
            if self.keypoints[idx].shape[0]: global_keypoints = np.vstack((global_keypoints, self.keypoints[idx]))
            if len(self.descriptors[idx]): global_descriptors = np.vstack((global_descriptors, self.descriptors[idx]))
        
        local_keypoints = local_keypoints[1:]
        local_descriptors = local_descriptors[1:]
        global_keypoints = global_keypoints[1:]
        global_descriptors = global_descriptors[1:]

        threshold = 30.0
        distance_matrix = cdist(local_descriptors, global_descriptors)
        matched_indices = np.where((distance_matrix <= threshold))
        # matched_indices = np.where((distance_matrix <= threshold) & (distance_matrix != 0.0))
        matched_keypoints1 = matched_indices[0]
        matched_keypoints2 = matched_indices[1]

        self.matching_line = Marker()
        for i in range(len(matched_keypoints1)):
            self.matching_line.header.frame_id = "/camera_init"
            self.matching_line.type = Marker.LINE_LIST
            self.matching_line.action = Marker.ADD
            line = Point(x=local_keypoints[matched_keypoints1[i]][0], y=local_keypoints[matched_keypoints1[i]][1], z=local_keypoints[matched_keypoints1[i]][2])
            self.matching_line.points.append(line)
            line = Point(x=global_keypoints[matched_keypoints2[i]][0], y=global_keypoints[matched_keypoints2[i]][1], z=global_keypoints[matched_keypoints2[i]][2])
            self.matching_line.points.append(line)
            self.matching_line.colors.append(ColorRGBA(1.0,0,0,1.0))
            self.matching_line.colors.append(ColorRGBA(1.0,0,0,1.0))
            self.matching_line.scale.x = 0.01

        # self.pub_matching_line.publish(self.matching_line)

        print(matched_keypoints1)
        print(matched_keypoints2)
    
    def make_map(self):        
        #slam_keypoints
        self.keypoints_msg = PointCloud2()
        self.keypoints_local_msg = PointCloud2()
        keypoints_msg_pc = o3d.geometry.PointCloud()
        keypoints_msg_pc_local = o3d.geometry.PointCloud()
        
        for pc_idx in range(self.local_graph_range[1]):
            pc_add = o3d.geometry.PointCloud()
            pc_add.points = o3d.utility.Vector3dVector(self.keypoints[pc_idx])

            if pc_idx >= self.local_graph_range[0]: keypoints_msg_pc_local += pc_add
            else: keypoints_msg_pc += pc_add
               
        self.keypoints_msg = orh.o3dpc_to_rospc(keypoints_msg_pc)
        self.keypoints_msg.header.frame_id = "/camera_init"
        self.keypoints_local_msg = orh.o3dpc_to_rospc(keypoints_msg_pc_local)
        self.keypoints_local_msg.header.frame_id = "/camera_init"
        
        self.pub_SLAM_keypoints.publish(self.keypoints_msg)
        self.pub_SLAM_keypoints_local.publish(self.keypoints_local_msg)
        
        # slam_poses
        self.keyposes_msg = PoseArray()
        for pose in self.poses:
            odom_msg_chach = Pose()
            odom_msg_chach.position.x = pose[0]
            odom_msg_chach.position.y = pose[1]
            odom_msg_chach.position.z = pose[2]
            odom_msg_chach.orientation.x = pose[3]
            odom_msg_chach.orientation.y = pose[4]
            odom_msg_chach.orientation.z = pose[5]
            odom_msg_chach.orientation.w = pose[6]
            self.keyposes_msg.poses.append(odom_msg_chach)
        self.keyposes_msg.header.frame_id = "/camera_init"
        self.pub_SLAM_poses.publish(self.keyposes_msg)

        # slam_map
        self.map_msg = PointCloud2()
        map_msg_pc = o3d.geometry.PointCloud()
        for pc in self.dense_scans:
            pc_chach = o3d.geometry.PointCloud()
            pc_chach.points = o3d.utility.Vector3dVector(np.array(pc))
            map_msg_pc += pc_chach
        self.map_msg = orh.o3dpc_to_rospc(map_msg_pc.voxel_down_sample(voxel_size=0.3))
        self.map_msg.header.frame_id = "/camera_init"
        self.pub_SLAM_map.publish(self.map_msg)
        
    def pub_map(self):
        self.pub_SLAM_keypoints.publish(self.keypoints_msg)
        self.pub_SLAM_keypoints_local.publish(self.keypoints_local_msg)
        self.pub_SLAM_poses.publish(self.keyposes_msg)
        self.pub_SLAM_map.publish(self.map_msg)
        print("pub complite")
    
    def get_data(self):
        kp0 = np.empty((0, 3))
        kp1 = np.empty((0, 3))
        pc0 = o3d.geometry.PointCloud()
        pc1 = o3d.geometry.PointCloud()
        
        for pc_idx in range(self.local_graph_range[1]):
            pc_chach = o3d.geometry.PointCloud()
            pc_chach.points = o3d.utility.Vector3dVector(np.array(self.dense_scans[pc_idx]))

            if pc_idx >= self.local_graph_range[0]:
                # kp1 += self.keypoints[pc_idx]
                kp1 = np.concatenate((kp1, self.keypoints[pc_idx]), axis=0)
                pc1 += pc_chach
            else: 
                kp0 = np.concatenate((kp0, self.keypoints[pc_idx]), axis=0)
                pc0 += pc_chach
        
        pc0 = pc0.voxel_down_sample(voxel_size=0.2)
        pc1 = pc1.voxel_down_sample(voxel_size=0.2)

        

        pc0 = np.array(pc0.points)
        pc1 = np.array(pc1.points)

        print(pc0.shape, pc1.shape)

        pc0_ones = np.ones((pc0.shape[0], 1))
        pc1_ones = np.ones((pc1.shape[0], 1))

        print(pc0.shape, pc1.shape)

        pc0 = np.concatenate((pc0, pc0_ones), axis=1)[:-(pc0.size % 8)]
        pc1 = np.concatenate((pc1, pc1_ones), axis=1)[:-(pc1.size % 8)]

        pc0 = pc0.reshape((-1, 8))
        pc1 = pc1.reshape((-1, 8))

        print(pc0.shape, pc1.shape)

        

        kp0_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in kp0]) 
        kp1_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in kp1])

        kp0_np = torch.tensor(kp0_np, dtype=torch.double)
        kp1_np = torch.tensor(kp1_np, dtype=torch.double)

        kp0_np = kp0_np[:, :3]
        kp1_np = kp1_np[:, :3]

        scores0 = np.ones_like(kp0_np[:,:1])
        scores1 = np.ones_like(kp1_np[:,:1])

        pc0, pc1 = torch.tensor(pc0, dtype=torch.double), torch.tensor(pc1, dtype=torch.double)

        #   File "/home/vision/ADD_prj/MDGAT-matcher/models/mdgat.py", line 118, in forward
            # B, _, _ = xyz.shape
            # ValueError: not enough values to unpack (expected 3, got 2)
        # 라는 오류가 나는데, 이건 원래 배치가 들어가면 앞에 배치사이즈가 나와야 되는데, 그게 안나와서 그런거임. 즉 배치가 1이어도 앞에 배치사이즈가 나와야함.

        kp0_np = np.expand_dims(kp0_np, axis=0)
        kp0_tensor = torch.tensor(kp0_np, dtype=torch.double)
        kp1_np = np.expand_dims(kp1_np, axis=0)
        kp1_tensor = torch.tensor(kp1_np, dtype=torch.double)
        # scores0 = np.expand_dims(scores0, axis=0)
        scores0_tensor = torch.tensor(scores0, dtype=torch.double)
        # scores1 = np.expand_dims(scores1, axis=0)
        scores1_tensor = torch.tensor(scores1, dtype=torch.double)
        pc0 = pc0.unsqueeze(0)
        pc1 = pc0.unsqueeze(0)


        return{
            'keypoints0': kp0_tensor,
            'keypoints1': kp1_tensor,
            'cloud0': pc0,
            'cloud1': pc1,
            'scores0' : scores0_tensor,
            'scores1' : scores1_tensor
        }

def handle_sigint(signal, frame):
    print("\n ---cancel by user---")
    sys.exit(0)

def model_inference(net, data, device):
    net.eval()
    with torch.no_grad():
        data['keypoints0'] = data['keypoints0'].to(device)
        data['keypoints1'] = data['keypoints1'].to(device)
        data['cloud0'] = data['cloud0'].to(device)
        data['cloud1'] = data['cloud1'].to(device)
        data['scores0'] = data['cloud0'].to(device)
        data['scores1'] = data['cloud1'].to(device)

        output = net(data)
        return output

if __name__ == '__main__':
    opt = parser.parse_args()
    rospy.init_node('kitti_dataset_setter')
    signal.signal(signal.SIGINT, handle_sigint)

    # set dataset
    Data = dataset(args=opt)
    pose = input("set new pose idx >> ")
    Data.set_current_pose_idx(int(pose))
    Data.make_map()

    # model setting
    path_checkpoint = opt.resume_model  
    checkpoint = torch.load(path_checkpoint, map_location={'cuda:2':'cuda:0'})  
    lr = checkpoint['lr_schedule']
    config = {
        'net': {
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
            # 'lr': lr,
            'lr': opt.learning_rate,
            'loss_method': opt.loss_method,
            'k': opt.k,
            'descriptor': opt.descriptor,
            'mutual_check': opt.mutual_check,
            'triplet_loss_gamma': opt.triplet_loss_gamma,
            'train_step':opt.train_step,
            'L':opt.l
        }
    }

    print("config: ", config)

    net = MDGAT(config.get('net', {}))
    optimizer = torch.optim.Adam(net.parameters(), lr=config.get('net', {}).get('lr'))
    net = torch.nn.DataParallel(net)
    net.load_state_dict(checkpoint['net'])
    
    if torch.cuda.is_available():
        # torch.cuda.set_device(opt.local_rank)
        device=torch.device('cuda:{}'.format(opt.local_rank))
    else:
        device = torch.device("cpu")
        print("### CUDA not available ###")
    
    print("model loaded")

    net.to(device)

    pred = Data.get_data()
    print(pred.keys())
    print(pred['keypoints0'].shape, pred['keypoints1'].shape, pred['cloud0'].shape, pred['cloud1'].shape, pred['scores0'].shape, pred['scores1'].shape)
    print("get data complite")

    data = model_inference(net, pred, device)

    print(data)

    while True:
        pose = input("map massage publist again? or set new pose idx >> ")
        if pose != "":
            Data.set_current_pose_idx(int(pose))
            Data.make_map()
            # Data._keypoints_l2_matching()
        Data.pub_map()