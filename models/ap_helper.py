""" Helper functions and class to calculate Average Precisions for 3D object detection.
"""
import os
import sys
import numpy as np
import torch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from eval_det import eval_det_multiprocessing, get_iou_obb
from my_util import nms_faster, softmax

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
    # B, num_proposal(32), 3
    pred_center = end_points['center']
    num_heading_bin = round(end_points['heading_scores'].shape[2]/3)
    # B, num_proposal
    pred_heading_class0 = torch.argmax(end_points['heading_scores'][:,:,0:num_heading_bin], -1)
    pred_heading_class1 = torch.argmax(end_points['heading_scores'][:,:,num_heading_bin:num_heading_bin*2], -1)
    pred_heading_class2 = torch.argmax(end_points['heading_scores'][:,:,num_heading_bin*2:num_heading_bin*3], -1)
    # (B,num_proposal,1) rad
    pred_heading_residual0 = torch.gather(end_points['heading_residuals'][:,:,0:num_heading_bin], 2, pred_heading_class0.unsqueeze(-1))
    pred_heading_residual1 = torch.gather(end_points['heading_residuals'][:,:,num_heading_bin:num_heading_bin*2], 2, pred_heading_class1.unsqueeze(-1))
    pred_heading_residual2 = torch.gather(end_points['heading_residuals'][:,:,num_heading_bin*2:num_heading_bin*3], 2, pred_heading_class2.unsqueeze(-1))
    # (B,num_proposal)
    pred_heading_residual0.squeeze_(2)
    pred_heading_residual1.squeeze_(2)
    pred_heading_residual2.squeeze_(2)

    bsize = pred_center.shape[0]  # B
    K = pred_center.shape[1]      # num_proposal
    pred_pose = np.zeros((bsize, K, 6))  # x,y,z,ori
    pred_center = pred_center.detach().cpu().numpy()
    for i in range(bsize):
        for j in range(K):
            heading_angle0 = config_dict['dataset_config'].class2angle(\
                pred_heading_class0[i,j].detach().cpu().numpy(), pred_heading_residual0[i,j].detach().cpu().numpy())
            heading_angle1 = config_dict['dataset_config'].class2angle(\
                pred_heading_class1[i,j].detach().cpu().numpy(), pred_heading_residual1[i,j].detach().cpu().numpy())
            heading_angle2 = config_dict['dataset_config'].class2angle(\
                pred_heading_class2[i,j].detach().cpu().numpy(), pred_heading_residual2[i,j].detach().cpu().numpy())
            
            pred_pose[i,j] = np.hstack([pred_center[i,j,:],heading_angle0,heading_angle1,heading_angle2])

    print(pred_pose)
    # (B,num_proposal,2)
    obj_logits = end_points['objectness_scores'].detach().cpu().numpy()
    # (B,num_proposal)
    obj_prob = softmax(obj_logits)[:,:,1]
    print(obj_prob)
    # use NMS plan A: only use the max prob
    pred_mask = np.zeros((bsize, K))
    for i in range(bsize):
        # probability
        boxes_with_prob = np.zeros((K,1))
        for j in range(K):
            boxes_with_prob[j,0] = obj_prob[i,j]
        nonempty_box_inds = np.array(range(K))  # [0,1,...,K-1]
        pick = nms_faster(boxes_with_prob)
        assert(len(pick)>0), "found no target..."
        # B, num_proposal
        pred_mask[i, nonempty_box_inds[pick]] = 1
    end_points['pred_mask'] = pred_mask

    # 根据 pred_mask，保存结果
    batch_pred_map_cls = []
    for i in range(bsize):
        cur_list = [(obj_prob[i,j], pred_pose[i,j]) \
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
    center_label = end_points['center_label']  # (B,1,3) xyz
    heading_class_label0 = end_points['heading_class_label'][:,0].unsqueeze(-1)        # (B,1)
    heading_class_label1 = end_points['heading_class_label'][:,1].unsqueeze(-1)        # (B,1)
    heading_class_label2 = end_points['heading_class_label'][:,2].unsqueeze(-1)        # (B,1)
    heading_residual_label0 = end_points['heading_residual_label'][:,0].unsqueeze(-1)  # (B,1)
    heading_residual_label1 = end_points['heading_residual_label'][:,1].unsqueeze(-1)  # (B,1)
    heading_residual_label2 = end_points['heading_residual_label'][:,2].unsqueeze(-1)  # (B,1)

    bsize = center_label.shape[0]  # B
    K2 = center_label.shape[1]     # 1
    # 存入 batch_gt_map_cls
    batch_gt_map_cls = []
    for i in range(bsize):
        for j in range(K2):
            heading_angle0 = config_dict['dataset_config'].class2angle(\
                heading_class_label0[i,j].detach().cpu().numpy(), heading_residual_label0[i,j].detach().cpu().numpy())
            heading_angle1 = config_dict['dataset_config'].class2angle(\
                heading_class_label1[i,j].detach().cpu().numpy(), heading_residual_label1[i,j].detach().cpu().numpy())
            heading_angle2 = config_dict['dataset_config'].class2angle(\
                heading_class_label2[i,j].detach().cpu().numpy(), heading_residual_label2[i,j].detach().cpu().numpy())
            for repeat in range(end_points['center'][1]):  # proposal
                batch_gt_map_cls.append([center_label[i,j,0], center_label[i,j,1],\
                    center_label[i,j,2], heading_angle0, heading_angle1, heading_angle2])

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
        
        bsize = len(batch_pred_map_cls)  # B
        assert(bsize == len(batch_gt_map_cls)), "batch_gt_map_cls size error!"
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]      # x,y,z,euler
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]  # prob,x,y,z,euler
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
