""" synthetic data loader.
An oriented bounding box is parameterized in pc coordinate
(Z upward, Y forward, X right), heading angle (from +X rotating to -Y)

Point clouds are in pc coordinate
Return heading class, heading residual for 3D bounding boxes.
"""

import os
import sys
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio  # to load .mat files for depth points
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)  # votenet
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
from pc_util import read_ply
from data_config import DatasetConfig

DC = DatasetConfig()  # dataset specific config

class SyntheticDatasetNew(Dataset):
    def __init__(self, split_set='demo', augment=False):

        self.data_path = os.path.join(ROOT_DIR, 'data/data_%s/ply'%(split_set))
        
        # ['006975', '007770', ...] 从低到高排序
        self.scan_names = sorted(list(set([os.path.basename(x)[0:4] \
            for x in os.listdir(self.data_path)])))

        self.augment = augment        # False
    
    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3)
            center_label: (1,3) for GT box center XYZ
            heading_class_label: (1) with int values in 0,...,NUM_HEADING_BIN-1
            heading_residual_label: (1)
            vote_label: (N,3) with votes XYZ
            vote_label_mask: (N,) with 0/1 with 1 indicating the point
                is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
        """
        scan_name = self.scan_names[idx]
        # N,3 (x,y,z)
        point_cloud = read_ply(os.path.join(self.data_path, scan_name+'.ply'))

        pcd_num = 2000
        if point_cloud.shape[0] >= pcd_num:
            choice = np.random.choice(point_cloud.shape[0], pcd_num, replace=False)
            point_cloud = point_cloud[choice, :]
        else:
            print('too little points...')

        # assert(point_cloud.shape[0] == 4000), "point cloud size error!"

        # pc coordinate (x right,y forward,z upward)
        point_cloud = point_cloud[:,0:3]

        angle_classes = np.zeros((3))
        angle_residuals = np.zeros((3))
        target_bboxes = np.zeros((1, 3))
        point_votes = np.zeros((point_cloud.shape[0],4))
        point_votes_mask = point_votes[:,0]   # bool (N)
        point_votes = point_votes[:,1:4]      # dx,dy,dz (N,3)

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:,0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)  # 场景 id 号
        ret_dict['scan_name'] = np.array(self.scan_names[idx]).astype(np.int64)

        return ret_dict

