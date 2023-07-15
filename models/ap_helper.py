""" Helper functions and class to calculate Average Precisions for 3D object detection.
"""
import os
import sys
import numpy as np
import torch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from eval_det import eval_det_cls, eval_det_multiprocessing, get_iou_obb
from my_util import nms_2d_faster, softmax
from box_util import get_3d_box

def flip_axis_to_camera(pc):
    '''从点云坐标系（右前上）转换到相机坐标系（右下前）'''
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]]
    pc2[...,1] *= -1
    return pc2

def flip_axis_to_depth(pc):
    '''从相机坐标系（右下前）转换到点云坐标系（右前上）'''
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]]
    pc2[...,2] *= -1
    return pc2

def parse_predictions(end_points, config_dict):
    """ Parse predictions to OBB parameters and suppress overlapping boxes
    
    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, nms_iou, conf_thresh}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """

    # 读取网络输出结果
    # B, num_proposal(256), 3
    pred_center = end_points['center']
    # B, num_proposal
    pred_heading_class = torch.argmax(end_points['heading_scores'], -1)
    # B, num_proposal, 1
    pred_heading_residual = torch.gather(end_points['heading_residuals'], 2,
        pred_heading_class.unsqueeze(-1))
    pred_heading_residual.squeeze_(2)
    # box_size
    box_size = config_dict['dataset_config'].box_size

    bsize = pred_center.shape[0]  # B
    K = pred_center.shape[1]      # num_proposal
    pred_corners_3d_upright_camera = np.zeros((bsize, K, 8, 3))
    pred_pose = np.zeros((bsize, K, 4))  # x,y,z,ori
    # 将 center 转换到相机坐标系下
    pred_center = pred_center.detach().cpu().numpy()
    pred_center_upright_camera = flip_axis_to_camera(pred_center)
    for i in range(bsize):
        for j in range(K):
            heading_angle = config_dict['dataset_config'].class2angle(\
                pred_heading_class[i,j].detach().cpu().numpy(), pred_heading_residual[i,j].detach().cpu().numpy())
             # 相机坐标系下 box 8 个角点
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, pred_center_upright_camera[i,j,:])
            pred_corners_3d_upright_camera[i,j] = corners_3d_upright_camera
            pred_pose[i,j] = np.hstack([pred_center[i,j,:],heading_angle])

    # B, num_proposal, 2
    obj_logits = end_points['objectness_scores'].detach().cpu().numpy()
    # B, num_proposal
    obj_prob = softmax(obj_logits)[:,:,1]
    # use 2d nms
    pred_mask = np.zeros((bsize, K))
    for i in range(bsize):
        # minx, minz, maxx, maxz, probability
        boxes_2d_with_prob = np.zeros((K,5))
        for j in range(K):
            boxes_2d_with_prob[j,0] = np.min(pred_corners_3d_upright_camera[i,j,:,0])
            boxes_2d_with_prob[j,2] = np.max(pred_corners_3d_upright_camera[i,j,:,0])
            boxes_2d_with_prob[j,1] = np.min(pred_corners_3d_upright_camera[i,j,:,2])
            boxes_2d_with_prob[j,3] = np.max(pred_corners_3d_upright_camera[i,j,:,2])
            boxes_2d_with_prob[j,4] = obj_prob[i,j]
        nonempty_box_inds = np.array(range(K))  # [0,1,2,K-1]
        pick = nms_2d_faster(boxes_2d_with_prob, config_dict['nms_iou'])
        assert(len(pick)>0)
        # B, num_proposal
        pred_mask[i, nonempty_box_inds[pick]] = 1
    end_points['pred_mask'] = pred_mask


    # 根据 pred_mask，保存 bbox 的 8 个角点
    batch_pred_map_cls = []
    for i in range(bsize):
        cur_list = [(pred_corners_3d_upright_camera[i,j], obj_prob[i,j], pred_pose[i,j]) \
            for j in range(K) if pred_mask[i,j]==1 and obj_prob[i,j]>config_dict['conf_thresh']]
        batch_pred_map_cls.append(cur_list)
    end_points['batch_pred_map_cls'] = batch_pred_map_cls

    return batch_pred_map_cls

def parse_groundtruths(end_points, config_dict):
    """
    Args:
        end_points: dict
            {center_label, heading_class_label, heading_residual_label,
            size_class_label, size_residual_label, sem_cls_label,
            box_label_mask}
        config_dict: dict
            {dataset_config}

    Returns:
        batch_gt_map_cls: bbox 的 8 个角点在相机坐标系下的坐标
    """
    center_label = end_points['center_label']
    heading_class_label = end_points['heading_class_label']
    heading_residual_label = end_points['heading_residual_label']
    box_size = config_dict['dataset_config'].box_size
    bsize = center_label.shape[0]
    K2 = center_label.shape[1]  # MAX_NUM_OBJ(1)
    gt_corners_3d_upright_camera = np.zeros((bsize, K2, 8, 3))
    # 将 center 转换到相机坐标系下
    gt_center_upright_camera = flip_axis_to_camera(center_label[:,:,0:3].detach().cpu().numpy())
    for i in range(bsize):
        for j in range(K2):
            heading_angle = config_dict['dataset_config'].class2angle(\
                heading_class_label[i,j].detach().cpu().numpy(), heading_residual_label[i,j].detach().cpu().numpy())
            # 相机坐标系下 box 8 个角点
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, gt_center_upright_camera[i,j,:])
            gt_corners_3d_upright_camera[i,j] = corners_3d_upright_camera

    # 存入 batch_gt_map_cls
    batch_gt_map_cls = []
    for i in range(bsize):
        batch_gt_map_cls.append([(gt_corners_3d_upright_camera[i,j]) \
            for j in range(gt_corners_3d_upright_camera.shape[1])])
    end_points['batch_gt_map_cls'] = batch_gt_map_cls

    return batch_gt_map_cls

class APCalculator(object):
    ''' Calculating Average Precision '''
    def __init__(self, ap_iou_thresh=0.25):
        """
        Args:
            ap_iou_thresh: float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.reset()
        
    def step(self, batch_pred_map_cls, batch_gt_map_cls):
        """ Accumulate one batch of prediction and groundtruth.
        
        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """
        
        bsize = len(batch_pred_map_cls)
        assert(bsize == len(batch_gt_map_cls))
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i] 
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i] 
            self.scan_cnt += 1
    
    def compute_metrics(self):
        """ compute Average Precision.
        """
        rec, _, ap = eval_det_multiprocessing(self.pred_map_cls, self.gt_map_cls, ovthresh=self.ap_iou_thresh, get_iou_func=get_iou_obb)
        
        ret_dict = {} 
        ret_dict['Average Precision'] = ap

        rec_list = []
        try:
            ret_dict['Recall'] = rec[-1]
            rec_list.append(rec[-1])
        except:
            ret_dict['Recall'] = 0
            rec_list.append(0)

        ret_dict['AR'] = np.mean(rec_list)
        return ret_dict

    def reset(self):
        self.gt_map_cls = {}    # {scan_id(B*iterations): [(bbox)]}
        self.pred_map_cls = {}  # {scan_id(B*iterations): [(bbox,score)]}
        self.scan_cnt = 0
