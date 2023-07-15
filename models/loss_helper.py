import torch
import torch.nn as nn
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from my_util import nn_distance, huber_loss

# FAR_THRESHOLD = 0.6
# NEAR_THRESHOLD = 0.3
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness

def compute_vote_loss(end_points):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
    
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.
    """

    # 读取霍夫投票的输出
    batch_size = end_points['seed_xyz'].shape[0]  # B(8), num_seed(128), 3
    num_seed = end_points['seed_xyz'].shape[1]
    vote_xyz = end_points['vote_xyz']             # B, num_seed, 3
    seed_inds = end_points['seed_inds'].long()    # B, num_seed

    # 读取 groundtruth
    # B, num_seed (bool)
    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds)
    # B, num_seed, 3 (repeat 3 times)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3)
    # B, num_seed, 3 (dx, dy, dz)
    seed_gt_votes = torch.gather(end_points['vote_label'], 1, seed_inds_expand)
    # B, num_seed, 3 (x, y, z)
    seed_gt_votes += end_points['seed_xyz']

    # 计算 vote_xyz 的 loss
    # B*num_seed, 1, 3
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, 1, 3)
    # B*num_seed, 1, 3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, 1, 3)
    # B*num_seed, 1
    _, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    # B, num_seed
    votes_dist = dist2.view(batch_size, num_seed)
    # average loss
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss

def compute_objectness_loss(end_points):
    """ 反映的是 proposal 对预测的自信程度.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
    """ 
    # 计算 aggregated_vote_xyz 的 loss
    aggregated_vote_xyz = end_points['aggregated_vote_xyz']  # (B(8),num_proposal(32),3) 点云坐标
    gt_center = end_points['center_label'][:,:,0:3]          # (B,1,3) xyz
    B = gt_center.shape[0]            # B
    K = aggregated_vote_xyz.shape[1]  # num_proposal
    # B, num_proposal
    dist1, _, _, _ = nn_distance(aggregated_vote_xyz, gt_center)

    # 计算 label 和 mask
    vote_inds = end_points['aggregated_vote_inds'].long()                         # (B,num_proposal)
    seed_inds = torch.gather(end_points['seed_inds'],1,vote_ind.long()).long()    # 全局索引
    objectness_label = torch.gather(end_points['vote_label_mask'], 1, seed_inds)  # (8,num_proposal)

    # 计算 objectness loss
    # B, num_proposal, 2
    objectness_scores = end_points['objectness_scores']  # (B,num_proposal,2)
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss)/(B*K)

    return objectness_loss, objectness_label

def compute_box_loss(end_points, config):
    """ Compute 3D bounding box loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
    """

    # compute center loss
    pred_center = end_points['center']  # (B,num_proposal,3) xyz
    batch_size = pred_center.shape[0]
    num_proposal = pred_center.shape[1]
    gt_center = end_points['center_label'][:,:,0:3]  # (B,1,3) xyz
    # dist1: (B,num_proposal), dist2: (B,1)
    dist1, _, dist2, _ = nn_distance(pred_center, gt_center)
    objectness_label = end_points['objectness_label'].float()  # (8,num_proposal)
    # 离 center 近的 proposal 的预测必须要准
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    # 每个物体都必须有预测
    centroid_reg_loss2 = torch.sum(dist2)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # compute heading class loss
    num_heading_bin = config.num_heading_bin  # 24
    # B, num_proposal
    heading_class_label0 = end_points['heading_class_label'][:,0].view(batch_size,1).repeat(1,num_proposal)
    heading_class_label1 = end_points['heading_class_label'][:,1].view(batch_size,1).repeat(1,num_proposal)
    heading_class_label2 = end_points['heading_class_label'][:,2].view(batch_size,1).repeat(1,num_proposal)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss0 = criterion_heading_class(end_points['heading_scores'][:,:,0:num_heading_bin].transpose(2,1), heading_class_label0)
    heading_class_loss1 = criterion_heading_class(end_points['heading_scores'][:,:,num_heading_bin:num_heading_bin*2].transpose(2,1), heading_class_label1)
    heading_class_loss2 = criterion_heading_class(end_points['heading_scores'][:,:,num_heading_bin*2:num_heading_bin*3].transpose(2,1), heading_class_label2)
    # 离 center 近的 proposal 的预测必须要准
    heading_class_loss0 = torch.sum(heading_class_loss0 * objectness_label)/(torch.sum(objectness_label)+1e-6)
    heading_class_loss1 = torch.sum(heading_class_loss1 * objectness_label)/(torch.sum(objectness_label)+1e-6)
    heading_class_loss2 = torch.sum(heading_class_loss2 * objectness_label)/(torch.sum(objectness_label)+1e-6)
    heading_class_loss = (heading_class_loss0+heading_class_loss1+heading_class_loss2)/3.0

    # compute heading residual loss
    # B, num_proposal
    heading_residual_label0 = end_points['heading_residual_label'][:,0].view(batch_size,1).repeat(1,num_proposal)
    heading_residual_label1 = end_points['heading_residual_label'][:,1].view(batch_size,1).repeat(1,num_proposal)
    heading_residual_label2 = end_points['heading_residual_label'][:,2].view(batch_size,1).repeat(1,num_proposal)
    # 将 -7.5~7.5*np.pi/180 转换到 -1~1
    heading_residual_normalized_label0 = heading_residual_label0 / (np.pi/num_heading_bin)
    heading_residual_normalized_label1 = heading_residual_label1 / (np.pi/num_heading_bin)
    heading_residual_normalized_label2 = heading_residual_label2 / (np.pi/num_heading_bin)
    # B, num_proposal, num_heading_bin
    heading_label_one_hot0 = torch.cuda.FloatTensor(batch_size, num_proposal, num_heading_bin).zero_()
    heading_label_one_hot1 = torch.cuda.FloatTensor(batch_size, num_proposal, num_heading_bin).zero_()
    heading_label_one_hot2 = torch.cuda.FloatTensor(batch_size, num_proposal, num_heading_bin).zero_()
    # num_heading_bin 的行向量为 one-hot 形式.
    heading_label_one_hot0.scatter_(2, heading_class_label0.unsqueeze(-1), 1)
    heading_label_one_hot1.scatter_(2, heading_class_label1.unsqueeze(-1), 1)
    heading_label_one_hot2.scatter_(2, heading_class_label2.unsqueeze(-1), 1)
    # B, num_proposal, num_heading_bin 逐元素相乘，再对最后一维求和。
    heading_residual_normalized_loss0 = huber_loss(torch.sum(
        end_points['heading_residuals_normalized'][:,:,0:num_heading_bin]*heading_label_one_hot0, -1)
        - heading_residual_normalized_label0)
    heading_residual_normalized_loss1 = huber_loss(torch.sum(
        end_points['heading_residuals_normalized'][:,:,num_heading_bin:num_heading_bin*2]*heading_label_one_hot1, -1)
        - heading_residual_normalized_label1)
    heading_residual_normalized_loss2 = huber_loss(torch.sum(
        end_points['heading_residuals_normalized'][:,:,num_heading_bin*2:num_heading_bin*3]*heading_label_one_hot2, -1)
        - heading_residual_normalized_label2)
    # 离 center 近的 proposal 的预测必须要准
    heading_residual_normalized_loss0 = torch.sum(heading_residual_normalized_loss0*objectness_label)/(torch.sum(objectness_label)+1e-6)
    heading_residual_normalized_loss1 = torch.sum(heading_residual_normalized_loss1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    heading_residual_normalized_loss2 = torch.sum(heading_residual_normalized_loss2*objectness_label)/(torch.sum(objectness_label)+1e-6)
    heading_residual_normalized_loss = (heading_residual_normalized_loss0 + heading_residual_normalized_loss1 + heading_residual_normalized_loss2)/3.0

    return center_loss, heading_class_loss, heading_residual_normalized_loss

def get_loss(end_points, config):
    """ Loss functions

    Args:
        end_points: dict
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """

    # Vote loss
    vote_loss = compute_vote_loss(end_points)
    end_points['vote_loss'] = vote_loss

    # Obj loss
    objectness_loss, objectness_label = compute_objectness_loss(end_points)
    end_points['objectness_loss'] = objectness_loss
    end_points['objectness_label'] = objectness_label
    # end_points['objectness_mask'] = objectness_mask
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]  # 8*32
    # end_points['pos_ratio'] = \
    #     torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    # end_points['neg_ratio'] = \
    #     torch.sum(objectness_mask.float())/float(total_num_proposal) - end_points['pos_ratio']

    # Box loss
    center_loss, heading_cls_loss, heading_reg_loss = \
        compute_box_loss(end_points, config)
    end_points['center_loss'] = center_loss
    end_points['heading_cls_loss'] = heading_cls_loss
    end_points['heading_reg_loss'] = heading_reg_loss
    box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss
    end_points['box_loss'] = box_loss

    # Final loss function
    loss = vote_loss + 0.5*objectness_loss + box_loss
    loss *= 10
    end_points['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points['objectness_scores'], 2)  # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float())/(obj_pred_val.shape[0]*obj_pred_val.shape[1])
    end_points['obj_acc'] = obj_acc

    return loss, end_points
