#encoding: utf-8
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from load_data_Rops import SparseDataset
import os
import torch.multiprocessing
import time
from utils.utils_test import (calculate_error, plot_match)
from models.superglue import SuperGlue
from models.mdgat_Rops import MDGAT
from scipy.spatial.distance import cdist

torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(
    description='Point cloud matching training ',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--visualize', type=bool, default=False,
    help='Visualize the matches')

parser.add_argument(
    '--calculate_pose', type=bool, default=True,
    help='Registrate the point cloud using the matched point pairs and calculate the pose')

parser.add_argument(
    '--voxel_size', type=float, default=0.2,
    help='voxel_size')

parser.add_argument(
    '--sinkhorn_iterations', type=int, default=100,
    help='Number of Sinkhorn iterations')

parser.add_argument(
    '--learning_rate', type=int, default=0.001,  #0.0001
    help='Learning rate')

parser.add_argument(
    '--epoch', type=int, default=1000,
    help='Number of epoches')

parser.add_argument(
    '--memory_is_enough', type=bool, default=True, 
    help='If memory is enough, load all the data')
        
parser.add_argument(
    '--batch_size', type=int, default=128, #128
    help='Batch size')

parser.add_argument(
    '--local_rank', type=int, default=0,
    help='Gpu rank')

parser.add_argument(
    '--resume', type=bool, default=False, # True False
    help='Resuming from existing model')

parser.add_argument(
    '--net', type=str, default='mdgat', 
    help='Choose net structure : mdgat superglue')

parser.add_argument(
    '--loss_method', type=str, default='gap_loss',
    help='Choose loss function : superglue triplet_loss gap_loss')

parser.add_argument(
    '--mutual_check', type=bool, default=False,  # True False
    help='If perform mutual check')

parser.add_argument(
    '--k', type=int, default=[128, None, 128, None, 64, None, 64, None], 
    help='Mdgat structure. None means connect all the nodes.')

parser.add_argument(
    '--l', type=int, default=9, 
    help='Layers number of GNN')

parser.add_argument(
    '--descriptor', type=str, default='Rops', 
    help='Choose keypoint descriptor : FPFH pointnet pointnetmsg FPFH_gloabal FPFH_only')

parser.add_argument(
    '--keypoints', type=str, default='Hariss3D', 
    help='Choose keypoints : sharp USIP lessharp')

parser.add_argument(
    '--ensure_kpts_num', type=bool, default=False, 
    help='')

parser.add_argument(
    '--max_keypoints', type=int, default=-1,  #1024
    help='Maximum number of keypoints'
            ' (\'-1\' keeps all keypoints)')

parser.add_argument(
    '--dataset', type=str, default='denseKITTI',  
    help='Used dataset')

parser.add_argument(
    '--resume_model', type=str, default='./checkpoint/denseKITTI/mdgat-l9-gap_loss-Rops-05_16_15_44/nomutualcheck-mdgat-batch128-lr0.001-gap_loss-Rops-Hariss3D-05_16_15_44/best_model_epoch_83(val_loss0.0859988382121983).pth',
    help='Path to the resumed model')

parser.add_argument(
    '--train_path', type=str, default='./denseKITTI/', 
    help='Path to the directory of training scans.')

# tsf_256_FPFH_16384-512-k1k16-2d-nonoise
# default='./KITTI/keypoints/tsf_256_FPFH_nr05_dr125',
# default='./KITTI/keypoints/tsf_256_FPFH_16384-512-k1k16-2d-nonoise',
parser.add_argument(
    '--keypoints_path', type=str, default='./denseKITTI/keypoints',
    help='Path to the directory of kepoints.')

parser.add_argument(
    '--txt_path', type=str, default='./denseKITTI', 
    help='Path to the directory of pairs.')

parser.add_argument(
    '--model_out_path', type=str, default='./checkpoint',
    help='Path to the directory of output model')

parser.add_argument(
    '--match_threshold', type=float, default=0.2,
    help='SuperGlue match threshold')

parser.add_argument(
    '--threshold', type=float, default=0.5,
    help='Ground truth distance threshold')

parser.add_argument(
    '--triplet_loss_gamma', type=float, default=0.6,
    help='Threshold for triplet loss and gap loss')

parser.add_argument(
    '--train_step', type=int, default=3,  
    help='Training step when using pointnet: 1,2,3')

if __name__ == '__main__':
    opt = parser.parse_args()
    

    test_set = SparseDataset(opt, 'test')
    test_loader = torch.utils.data.DataLoader(dataset=test_set, shuffle=False, batch_size=1, num_workers=1, drop_last=True, pin_memory = True)
 
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
    if opt.net == 'superglue':
        net = SuperGlue(config.get('net', {}))
        print(opt.net)
    else:
        net = MDGAT(config.get('net', {}))
    optimizer = torch.optim.Adam(net.parameters(), lr=config.get('net', {}).get('lr'))
    net = torch.nn.DataParallel(net)
    net.load_state_dict(checkpoint['net']) 
    start_epoch = checkpoint['epoch'] + 1  
    best_loss = checkpoint['loss']
    print('Resume from ', opt.resume_model)

    
    if torch.cuda.is_available():
        # torch.cuda.set_device(opt.local_rank)
        device=torch.device('cuda:{}'.format(opt.local_rank))
    else:
        device = torch.device("cpu")
        print("### CUDA not available ###")

    net.to(device)
    
    '''
        model.eval():   will notify all your layers that you are in eval mode, 
                        that way, batchnorm or dropout layers will work in eval 
                        mode instead of training mode.
        torch.no_grad():impacts the autograd engine and deactivate it. It will 
                        reduce memory usage and speed up computations but you 
                        won’t be able to backprop (which you don’t want in an eval script).
    '''
    with torch.no_grad():
        mean_test_loss = []; precision_array = []; accuracy_array = []; recall_array = []
        trans_error_array = []; rot_error_array = []; relative_trans_error_array = []; relative_rot_error_array = []
        repeatibilty_array = []; valid_num_array = []; all_num_array = []; inlier_array = [] 
        kpnum_array = []; fp_rate_array = []; tp_rate_array = []; tp_rate2_array = []; inlier_ratio_array= [];tm_a=[];fm_a=[]
        fail = 0
        baned_data = 0
        
        for i, pred in enumerate(test_loader):
            ### eval ###
            begin = time.time()
            net.double().eval()                
            for k in pred:
                if k!='idx0' and k!='idx1' and k!='sequence':
                    if type(pred[k]) == torch.Tensor:
                        pred[k] = Variable(pred[k].cuda().detach())
                    else:
                        pred[k] = Variable(torch.stack(pred[k]).cuda().detach())
            
            data = net(pred) 
            pred = {**pred, **data}

            for b in range(len(pred['idx0'])):
                '''If you got KITTI dataset, load the point cloud for better visualization'''
                # pc0_path = os.path.join('/home/chenghao/Mount/Dataset/KITTI_odometry/preprocess-undownsample-n8', pred['sequence'][b], '%06d.bin'%pred['idx0'][b])
                # pc1_path = os.path.join('/home/chenghao/Mount/Dataset/KITTI_odometry/preprocess-undownsample-n8', pred['sequence'][b], '%06d.bin'%pred['idx1'][b])
                # pc0, pc1 = np.fromfile(pc0_path, dtype=np.float32), np.fromfile(pc1_path, dtype=np.float32)
                # pc0, pc1 = pc0.reshape(-1, 8), pc1.reshape(-1, 8)
                pc0, pc1 = [],[]

                kpts0, kpts1 = pred['keypoints0'][b].cpu().numpy(), pred['keypoints1'][b].cpu().numpy()
                idx = pred['idx0'][b]
                matches, matches1, conf = pred['matches0'][b].cpu().detach().numpy(), pred['matches1'][b].cpu().detach().numpy(), pred['matching_scores0'][b].cpu().detach().numpy()
                valid = matches > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]

                mconf = conf[valid]

                mutual0 = np.arange(len(matches))[valid] == matches1[matches[valid]]
                mutual0 = np.arange(len(matches))[valid][mutual0]
                mutual1 = matches[mutual0]
                x = np.ones(len(matches1)) == 1
                x[mutual1] = False
                valid1 = matches1 > -1
                # extrakpt1 = kpts1[valid1 & x]
                # extrakpt0 = kpts0[matches1[valid1 & x]]
                # mkpts0 = np.vstack((mkpts0, extrakpt0))
                # mkpts1 = np.vstack((mkpts1, extrakpt1))

                mconf = conf[valid]
                # mscores

                ## ground truth ##
                matches_gt, matches_gt1 = pred['gt_matches0'][b].cpu().detach().numpy(), pred['gt_matches1'][b].cpu().detach().numpy()
                matches_gt[matches_gt == len(matches_gt1)] = -1
                matches_gt1[matches_gt1 == len(matches_gt)] = -1
                valid_gt = matches_gt > -1

                valid_num = np.sum(valid_gt)
                # valid_num = pred['rep'][b].cpu().detach().numpy()
                all_num = len(valid_gt)
                repeatibilty = valid_num/all_num 
                repeatibilty_array.append(repeatibilty)

                if valid_gt.sum() < len(matches_gt)*0.1:
                    # print('not enough ground truth match, ban the pair')
                    baned_data+=1
                    fail+=1
                    continue

                mkpts0_gt = kpts0[valid_gt]
                mkpts1_gt = kpts1[matches_gt[valid_gt]]
                mutual0 = np.arange(len(matches_gt))[valid_gt] == matches_gt1[matches_gt[valid_gt]]
                # mutual0_inv = 1-mutual0
                mutual0 = np.arange(len(matches_gt))[valid_gt][mutual0]
                mutual1 = matches_gt[mutual0]
                x = np.ones(len(matches_gt1)) == 1
                x[mutual1] = False               
                valid_gt1 = matches_gt1 > -1
                # extrakpt1 = kpts1[valid_gt1 & x]
                # extrakpt0 = kpts0[matches_gt1[valid_gt1 & x]]
                # mkpts0_gt = np.vstack((mkpts0_gt, extrakpt0))
                # mkpts1_gt = np.vstack((mkpts1_gt, extrakpt1))


                mscores_gt = pred['scores0'][b].cpu().numpy()[valid_gt]
                gt_idx = np.arange(len(kpts0))[valid_gt]
            

                if len(mkpts0) < 4:
                    fail+=1
                    print('registration fail')
                else:
                    ''' calculate false positive ,true positive ,true nagetive, precision, accuracy, recall '''
                    true_positive = [(matches[i] == matches_gt[i]) and (valid[i]) for i in range(len(kpts0))]
                    true_negativate = [(matches[i] == matches_gt[i]) and not (valid[i]) for i in range(len(kpts0))]
                    false_positive = [valid[i] and (matches_gt[i]==-1) for i in range(len(kpts0))]
                    ckpts0 = kpts0[true_positive]
                    ckpts1 = [matches[true_positive]]
                    precision = np.sum(true_positive) / np.sum(valid) if np.sum(valid) > 0 else 0
                    recall = np.sum(true_positive) / np.sum(valid_gt) if np.sum(valid) > 0 else 0
                    tm = np.sum(true_positive) 
                    fm = np.sum(false_positive) 
                    matching_score = np.sum(true_positive) / len(kpts0) if len(kpts0) > 0 else 0
                    accuracy = (np.sum(true_positive) + np.sum(true_negativate))/len(matches_gt)
                    fp_rate = np.sum(false_positive)/np.sum(matches_gt==-1)
                    tp_rate = np.sum([valid[i] and (matches_gt[i]>-1) for i in range(len(kpts0))])/np.sum(matches_gt > -1)
                    tp_rate2 = np.sum(true_positive)/np.sum(matches_gt > -1)
                    
                    '''calculate pose error, inlier and failure rate'''
                    if opt.calculate_pose:
                        T, inlier, inlier_ratio, trans_error, rot_error = calculate_error(mkpts0, mkpts1, pred, b) 

                        if trans_error>2 or rot_error>5 or np.isnan(trans_error) or np.isnan(rot_error):
                            fail+=1
                            print('registration fail')
                        else:
                            precision_array.append(precision)
                            accuracy_array.append(accuracy)
                            recall_array.append(recall)
                            trans_error_array.append(trans_error)
                            rot_error_array.append(rot_error)
                            inlier_array.append(inlier)
                            inlier_ratio_array.append(inlier_ratio)
                            fp_rate_array.append(fp_rate)
                            tp_rate_array.append(tp_rate)
                            tp_rate2_array.append(tp_rate2)
                            tm_a.append(tm)
                            fm_a.append(fm)
                            # else:
                            #     baned_data+=1
                            print('idx{}, inlier {}, rep {:.3f}， inlier_ratio {:.3f}, precision {:.3f}, accuracy {:.3f}, recall {:.3f}, fp_rate {:.3f}, tp_rate {:.3f}, trans_error {:.3f}, rot_error {:.3f} '.format(
                                idx, inlier, repeatibilty,inlier_ratio, precision, accuracy, recall, fp_rate, tp_rate, trans_error, rot_error))
                    else:
                        T=[]
                        print('idx{}, precision {:.3f}, accuracy {:.3f}, recall {:.3f}, true match {:.3f}, false match {:.3f}, fp_rate {:.3f}, tp_rate {:.3f}'.format(
                            idx, precision, accuracy, recall,tm,fm, fp_rate, tp_rate))

                    if opt.visualize:
                        plot_match(pc0, pc1, kpts0, kpts1, mkpts0, mkpts1, mkpts0_gt, mkpts1_gt, matches, mconf, true_positive, false_positive, T, opt.vis_line_width)

                    

        precision_mean = np.mean(precision_array)
        accuracy_mean = np.mean(accuracy_array)
        recall_mean = np.mean(recall_array)
        trans_error_mean = np.mean(trans_error_array)
        rot_error_mean = np.mean(rot_error_array)
        repeatibilty_array_mean = np.mean(repeatibilty_array)
        inlier_mean = np.mean(inlier_array)
        inlier_ratio_mean = np.mean(inlier_ratio_array)
        fp_rate_mean = np.mean(fp_rate_array)
        tp_rate_mean = np.mean(tp_rate_array)
        tp_rate_mean2 = np.mean(tp_rate2_array)
        tm = np.mean(tm_a)
        fm = np.mean(fm_a)
        print('average repeatibility: {:.3f}, inlier_mean {:.3f}, inlier_ratio_mean {:.3f}, fail {:.6f}, precision_mean {:.3f}, accuracy_mean {:.3f}, recall_mean {:.3f}, true match {:.3f}, false match {:.3f}, fp_rate_mean {:.3f}, tp_rate_mean {:.3f}, tp_rate_mean2 {:.3f}, trans_error_mean {:.3f}, rot_error_mean {:.3f} '.format(
            repeatibilty_array_mean, inlier_mean, inlier_ratio_mean, fail/i, precision_mean, accuracy_mean, recall_mean,tm,fm, fp_rate_mean, tp_rate_mean, tp_rate_mean2, trans_error_mean, rot_error_mean ))
        # print('valid num {}, all_num {}'.format(valid_num_mean, all_num_mean))
        print('baned_data {}'.format(baned_data/i))
