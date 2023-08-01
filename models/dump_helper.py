import numpy as np
import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
from my_util import softmax

DUMP_CONF_THRESH = 0.5  # Dump boxes with obj prob larger than that.

def dump_results(end_points, dump_dir, config):
    ''' Dump results.
    Args:
        end_points: dict
            {..., pred_mask}
            pred_mask is a binary mask array of size (batch_size, num_proposal) computed by running NMS and empty box removal
    Returns:
        None
    '''
    if not os.path.exists(dump_dir):
        os.system('mkdir %s'%(dump_dir))

    # INPUT
    point_clouds = end_points['point_clouds'].cpu().numpy()
    batch_size = point_clouds.shape[0]

    # NETWORK OUTPUTS
    # B, num_seed(1024), 3
    seed_xyz = end_points['seed_xyz'].detach().cpu().numpy()
    # B, num_seed, 3
    vote_xyz = end_points['vote_xyz'].detach().cpu().numpy()
    # B, num_proposal(256), 3
    aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()
    # B, num_proposal, 2
    objectness_scores = end_points['objectness_scores'].detach().cpu().numpy()
    # B, num_proposal, 3
    pred_center = end_points['center'].detach().cpu().numpy()
    # B, num_proposal
    pred_heading_class = torch.argmax(end_points['heading_scores'], -1)
    # B, num_proposal, 1
    pred_heading_residual = torch.gather(end_points['heading_residuals'], 2, pred_heading_class.unsqueeze(-1))
    # B, num_proposal
    pred_heading_class = pred_heading_class.detach().cpu().numpy()
    # B, num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy()

    # OTHERS
    # B,num_proposal
    pred_mask = end_points['pred_mask']
    idx_beg = 0

    for i in range(batch_size):
        pc = point_clouds[i,:,:]
        # num_proposal,
        objectness_prob = softmax(objectness_scores[i,:,:])[:,1]

        # Dump various point clouds
        pc_util.write_ply(pc, os.path.join(dump_dir, '%06d_pc.ply'%(idx_beg+i)))
        pc_util.write_ply(seed_xyz[i,:,:], os.path.join(dump_dir, '%06d_seed_pc.ply'%(idx_beg+i)))
        pc_util.write_ply(vote_xyz[i,:,:], os.path.join(dump_dir, '%06d_vgen_pc.ply'%(idx_beg+i)))
        pc_util.write_ply(aggregated_vote_xyz[i,:,:], os.path.join(dump_dir, '%06d_aggregated_vote_pc.ply'%(idx_beg+i)))
        pc_util.write_ply(pred_center[i,:,0:3], os.path.join(dump_dir, '%06d_proposal_pc.ply'%(idx_beg+i)))
        if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
            pc_util.write_ply(pred_center[i,objectness_prob>DUMP_CONF_THRESH,0:3], os.path.join(dump_dir, '%06d_confident_proposal_pc.ply'%(idx_beg+i)))

        # Dump predicted bounding boxes
        if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
            num_proposal = pred_center.shape[1]
            obbs = []
            for j in range(num_proposal):
                obb = config.param2obb(pred_center[i,j,0:3], pred_heading_class[i,j], pred_heading_residual[i,j])
                obbs.append(obb)
            if len(obbs)>0:
                # (num_proposal, 7)
                obbs = np.vstack(tuple(obbs))
                pc_util.write_oriented_bbox(obbs[objectness_prob>DUMP_CONF_THRESH,:], os.path.join(dump_dir, '%06d_pred_confident_bbox.ply'%(idx_beg+i)))
                pc_util.write_oriented_bbox(obbs[np.logical_and(objectness_prob>DUMP_CONF_THRESH, pred_mask[i,:]==1),:], os.path.join(dump_dir, '%06d_pred_confident_nms_bbox.ply'%(idx_beg+i)))
                pc_util.write_oriented_bbox(obbs[pred_mask[i,:]==1,:], os.path.join(dump_dir, '%06d_pred_nms_bbox.ply'%(idx_beg+i)))
                pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, '%06d_pred_bbox.ply'%(idx_beg+i)))



