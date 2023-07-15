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
from data_config import DatasetConfig

DC = DatasetConfig()  # dataset specific config

class SyntheticDataset(Dataset):
    def __init__(self, split_set='train', use_height=False, augment=False):

        self.data_path = os.path.join(ROOT_DIR, 'data/data_%s'%(split_set))
        
        # ['006975', '007770', ...] 从低到高排序
        self.scan_names = sorted(list(set([os.path.basename(x)[0:6] \
            for x in os.listdir(self.data_path)])))

        self.augment = augment        # False
        self.use_height = use_height  # False
    
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
        point_cloud = np.load(os.path.join(self.data_path, scan_name)+'_pc.npz')['pc']
        # 1,6 (x,y,z,euler)
        bboxes = np.load(os.path.join(self.data_path, scan_name)+'_bbox.npy')
        # N,4 (bool,dx,dy,dz)
        point_votes = np.load(os.path.join(self.data_path, scan_name)+'_votes.npz')['point_votes']

        assert(512 < point_cloud.shape[0] < 10000)

        # pc coordinate (x right,y forward,z upward)
        point_cloud = point_cloud[:,0:3]
        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1)  # (N,4)

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:,0] = -1 * point_cloud[:,0]
                bboxes[:,0] = -1 * bboxes[:,0]
                bboxes[:,3] = np.pi - bboxes[:,3]
                point_votes[:,[1]] = -1 * point_votes[:,[1]]

            # Rotation along Z-axis
            rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
            rot_mat = pc_util.rotz(rot_angle)

            point_votes_end = np.zeros_like(point_votes)
            point_votes_end[:,1:4] = np.dot(point_cloud[:,0:3] + point_votes[:,1:4], np.transpose(rot_mat))

            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            bboxes[:,0:3] = np.dot(bboxes[:,0:3], np.transpose(rot_mat))
            bboxes[:,3] -= rot_angle
            point_votes[:,1:4] = point_votes_end[:,1:4] - point_cloud[:,0:3]

            # Augment point cloud scale: 0.95x-1.05x
            scale_ratio = np.random.random()*0.1+0.95
            scale_ratio = np.expand_dims(np.tile(scale_ratio,3),0)
            point_cloud[:,0:3] *= scale_ratio
            bboxes[:,0:3] *= scale_ratio
            point_votes[:,1:4] *= scale_ratio
            if self.use_height:
                point_cloud[:,-1] *= scale_ratio[0,0]

        # ------------------------------- LABELS ------------------------------
        angle_classes = np.zeros((3))
        angle_residuals = np.zeros((3))
        target_bboxes = np.zeros((1, 3))

        if np.shape(bboxes)[0] > 0:
            bbox = bboxes[0]
            angle_class, angle_residual = DC.angle2class(bbox[3])
            angle_classes[0] = angle_class
            angle_residuals[0] = angle_residual
            angle_class, angle_residual = DC.angle2class(bbox[4])
            angle_classes[1] = angle_class
            angle_residuals[1] = angle_residual
            angle_class, angle_residual = DC.angle2class(bbox[5])
            angle_classes[2] = angle_class
            angle_residuals[2] = angle_residual
            target_bboxes[0,:] = bbox[0:3]

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

        return ret_dict

def viz_votes(pc, point_votes, point_votes_mask):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_votes_mask==1)
    pc_obj = pc[inds,0:3]
    pc_obj_voted1 = pc_obj + point_votes[inds,0:3]
    pc_obj_voted2 = pc_obj + point_votes[inds,3:6]
    pc_obj_voted3 = pc_obj + point_votes[inds,6:9]
    pc_util.write_ply(pc_obj, 'pc_obj.ply')
    pc_util.write_ply(pc_obj_voted1, 'pc_obj_voted1.ply')
    pc_util.write_ply(pc_obj_voted2, 'pc_obj_voted2.ply')
    pc_util.write_ply(pc_obj_voted3, 'pc_obj_voted3.ply')

def viz_obb(pc, label, mask, angle_classes, angle_residuals,
    size_classes, size_residuals):
    """ Visualize oriented bounding box ground truth
    pc: (N,3)
    label: (K,3)  K == MAX_NUM_OBJ
    mask: (K,)
    angle_classes: (K,)
    angle_residuals: (K,)
    size_classes: (K,)
    size_residuals: (K,3)
    """
    oriented_boxes = []
    K = label.shape[0]
    for i in range(K):
        if mask[i] == 0: continue
        obb = np.zeros(7)
        obb[0:3] = label[i,0:3]
        heading_angle = DC.class2angle(angle_classes[i], angle_residuals[i])
        box_size = DC.class2size(size_classes[i], size_residuals[i])
        obb[3:6] = box_size
        obb[6] = -1 * heading_angle
        print(obb)
        oriented_boxes.append(obb)
    pc_util.write_oriented_bbox(oriented_boxes, 'gt_obbs.ply')
    pc_util.write_ply(label[mask==1,:], 'gt_centroids.ply')

if __name__=='__main__':
    d = SyntheticDataset(use_height=True, augment=True)
    sample = d[200]
    print(sample['vote_label'].shape, sample['vote_label_mask'].shape)
    pc_util.write_ply(sample['point_clouds'], 'pc.ply')
    viz_votes(sample['point_clouds'], sample['vote_label'], sample['vote_label_mask'])
    viz_obb(sample['point_clouds'], sample['center_label'], sample['box_label_mask'],
        sample['heading_class_label'], sample['heading_residual_label'],
        sample['size_class_label'], sample['size_residual_label'])
