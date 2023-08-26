""" Demo of using VoteNet 3D object detector to detect objects from a point cloud.
"""

import os
import sys
import numpy as np
import argparse
import importlib
import time

import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR  # votenet
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from ap_helper import parse_predictions
from pc_util import random_sampling, read_ply, write_ply
from synthetic_dataset import DC  # dataset config
import data_config
from autolab_core import Logger, RigidTransform
from scipy.spatial.transform import Rotation as R


LOG_FOUT = open(os.path.join(BASE_DIR, 'data/data_demo/results/log_train.txt'), 'a')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:,0:3]

    # floor_height = np.percentile(point_cloud[:,2],0.99)
    # height = point_cloud[:,2] - floor_height
    # point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1)  # (N,4)

    pcd_num = 2000
    assert(pcd_num <= point_cloud.shape[0]), "point cloud size error!"
    choice = np.random.choice(point_cloud.shape[0], pcd_num, replace=False)
    point_cloud = point_cloud[choice, :]

    pc = np.expand_dims(point_cloud.astype(np.float32), 0)  # (1,num_points,3)
    return pc

if __name__=='__main__':

    # Set file paths and dataset config
    demo_dir = os.path.join(BASE_DIR, 'data/data_demo') 
    checkpoint_path = os.path.join(demo_dir, 'checkpoint.tar')
    # ['006975', '007770', ...] 从低到高排序
    scan_names = sorted(list(set([os.path.basename(x)[0:4] \
        for x in os.listdir(os.path.join(demo_dir, 'ply'))])))
    gt_data = np.loadtxt(demo_dir + '/end_pose_ref.log')

    eval_config_dict = {'conf_thresh': 0.0005, 'dataset_config': DC}

    # Init the model and optimzier
    MODEL = importlib.import_module('votenet')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.device_count())
    net = MODEL.VoteNet(num_proposal=32, input_feature_dim=0,
        sampling='random', num_heading_bin=DC.num_heading_bin)
    net.to(device)
    print('Constructed model.')
    
    # Load checkpoint
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))
    net.eval()  # eval model will lead to bad predict, don't know why

    Tm_4 = RigidTransform(
        rotation=RigidTransform.z_axis_rotation((-150)*np.pi/180),  # dataset1 +45
        translation=np.array([0.0,0.0,0.0]),
        from_frame="marker_4",
        to_frame="base",
    )
    Tm_5 = RigidTransform(
        rotation=RigidTransform.x_axis_rotation(0),
        translation=np.array([0, 0, -4.5]),  #-(- 4.237 + 6.3 + 1.3)
        from_frame="marker_5",
        to_frame="marker_4",
    )

    error = np.zeros((len(scan_names),2))
    for i in range(len(scan_names)):

        scan_name = scan_names[i]

        T_qua2rota = RigidTransform(
            rotation=np.array([gt_data[int(scan_name)-1][3], gt_data[int(scan_name)-1][4], gt_data[int(scan_name)-1][5], gt_data[int(scan_name)-1][6]]),
            translation=np.array([gt_data[int(scan_name)-1][0], gt_data[int(scan_name)-1][1], gt_data[int(scan_name)-1][2]]),
            from_frame="marker_5",
            to_frame="world",
        )
        pose_gt = T_qua2rota*(Tm_4*Tm_5).inverse()

        # Load and preprocess input point cloud
        point_cloud = read_ply(demo_dir+'/ply/'+scan_name+'.ply')
        # point_cloud = np.load(demo_dir + '/000950_pc.npz')['pc']
        pc = preprocess_point_cloud(point_cloud)
        print('Loaded point cloud data: %s'%(scan_name))
    
        # Model inference
        inputs = {'point_clouds': torch.from_numpy(pc).to(device)}
        # tic = time.time()
        with torch.no_grad():
            end_points = net(inputs)
        # toc = time.time()
        # print('Inference time: %f'%(toc-tic))
        pred_map_cls = parse_predictions(end_points, eval_config_dict)  # prob,x,y,z,euler
        # print('Finished detection. %d object detected.'%(len(pred_map_cls[0])))
    
        # dump_dir = os.path.join(demo_dir, 'results/endo_%d'%dir_list[i])
        # if not os.path.exists(dump_dir): os.mkdir(dump_dir)
        # MODEL.dump_results(end_points, dump_dir, DC)
        # boxPoints = get_3d_box(DC.box_size, pred_map_cls[0][0][2][3], pred_map_cls[0][0][2][:3])
        # write_ply(boxPoints, os.path.join(dump_dir, 'bbox.ply'))
        # print('Dumped detection results to folder %s'%(dump_dir))

        # 计算误差
        position_error = np.sqrt(np.sum((pose_gt.translation-pred_map_cls[0][0][1][0:3])*(pose_gt.translation-pred_map_cls[0][0][1][0:3])))
        R_predict = R.from_euler('XYZ', pred_map_cls[0][0][1][3:6]).as_matrix()
        R_gt = pose_gt.rotation
        z_predict = R_predict[:,2]
        z_gt = R_gt[:,2]
        z_error = np.degrees(np.arccos(np.dot(z_predict, z_gt)))
        x_predict = R_predict[:,0]
        x_gt = R_gt[:,0]
        x_error = np.degrees(np.arccos(np.dot(x_predict, x_gt)))
        print(R.from_euler('XYZ', pred_map_cls[0][0][1][3:6]).as_euler('xyz',degrees=True))

        
        if len(pred_map_cls[0]) > 0:
            # error[i][0] = abs(pose_gt.translation[0]-pred_map_cls[0][0][1][0])
            # error[i][1] = abs(pose_gt.translation[1]-pred_map_cls[0][0][1][1])
            # error[i][2] = abs(pose_gt.translation[2]-pred_map_cls[0][0][1][2])
            # error[i][3] = position_error

            error[i][0] = z_error
            error[i][1] = x_error
            log_string('position errror: %f'%(position_error))
            log_string('z_axis error: %f'%(z_error))
            log_string('x_axis error: %f'%(x_error))
    np.save('error.npy', error)




