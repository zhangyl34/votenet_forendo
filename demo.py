""" Demo of using VoteNet 3D object detector to detect objects from a point cloud.
"""

import os
import sys
import numpy as np
import argparse
import importlib
import time

parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=40000, help='Point Number [default: 40000]')
FLAGS = parser.parse_args()

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


LOG_FOUT = open(os.path.join(BASE_DIR, 'data/data_demo/results/log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:,0:3]  # do not use color
    floor_height = np.percentile(point_cloud[:,2],0.99)
    height = point_cloud[:,2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1)  # (N,4)
    print(point_cloud.shape)
    if (FLAGS.num_point<point_cloud.shape[0]):
        point_cloud = random_sampling(point_cloud, FLAGS.num_point)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0)  # (1,40000,4)
    return pc

def get_3d_box(box_size, heading_angle, center):
    c = np.cos(-heading_angle)
    s = np.sin(-heading_angle)
    R = np.array([[c, -s,  0],
                  [s,  c,  0],
                  [0,  0,  1]])
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    z_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0]
    corners_3d[1,:] = corners_3d[1,:] + center[1]
    corners_3d[2,:] = corners_3d[2,:] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d

if __name__=='__main__':

    # Set file paths and dataset config
    demo_dir = os.path.join(BASE_DIR, 'data/data_demo') 
    checkpoint_path = os.path.join(demo_dir, 'checkpoint.tar')
    pc_paths = [os.path.join(demo_dir, 'rii6_0.ply')]
                # os.path.join(demo_dir, 'rii3.ply'),
                # os.path.join(demo_dir, 'rii4.ply'),
                # os.path.join(demo_dir, 'rii5.ply'),
                # os.path.join(demo_dir, 'rii6.ply'),
                # os.path.join(demo_dir, 'rii7.ply'),
                # os.path.join(demo_dir, 'rii8.ply'),
                # os.path.join(demo_dir, 'rii9.ply'),
                # os.path.join(demo_dir, 'rii10.ply'),
                # os.path.join(demo_dir, 'rii11.ply'),
                # os.path.join(demo_dir, 'rii12.ply'),
                # os.path.join(demo_dir, 'rii13.ply'),
                # os.path.join(demo_dir, 'rii14.ply'),
                # os.path.join(demo_dir, 'rii15.ply')]

    eval_config_dict = {'nms_iou': 0.25, 'conf_thresh': 0.15, 'dataset_config': DC}

    # Init the model and optimzier
    MODEL = importlib.import_module('votenet')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.device_count())
    net = MODEL.VoteNet(num_proposal=256, input_feature_dim=1,
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
    net.eval()  # set model to eval mode (for bn and dp)
    
    # dir_list=[2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    dir_list=[0]
    for i in range(len(pc_paths)):
        # Load and preprocess input point cloud
        point_cloud = read_ply(pc_paths[i])
        pc = preprocess_point_cloud(point_cloud)
        print('Loaded point cloud data: %s'%(pc_paths[i]))
    
        # Model inference
        inputs = {'point_clouds': torch.from_numpy(pc).to(device)}
        tic = time.time()
        with torch.no_grad():
            end_points = net(inputs)
        toc = time.time()
        print('Inference time: %f'%(toc-tic))
        end_points['point_clouds'] = inputs['point_clouds']
        pred_map_cls = parse_predictions(end_points, eval_config_dict)
        print('Finished detection. %d object detected.'%(len(pred_map_cls[0])))
    
        dump_dir = os.path.join(demo_dir, 'results/rii6_%d'%dir_list[i])
        if not os.path.exists(dump_dir): os.mkdir(dump_dir)
        # MODEL.dump_results(end_points, dump_dir, DC)
        boxPoints = get_3d_box(DC.box_size, pred_map_cls[0][0][2][3], pred_map_cls[0][0][2][:3])
        write_ply(boxPoints, os.path.join(dump_dir, 'bbox.ply'))
        print('Dumped detection results to folder %s'%(dump_dir))

        log_string('data rii6_%d:'%(dir_list[i]))
        log_string('Finished detection. %d object detected.'%(len(pred_map_cls[0])))
        if len(pred_map_cls[0]) > 0:
            log_string('result: x %f, y %f, z %f, ori %f'%(pred_map_cls[0][0][2][0],
                pred_map_cls[0][0][2][1],pred_map_cls[0][0][2][2],pred_map_cls[0][0][2][3]))


