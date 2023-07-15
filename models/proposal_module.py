import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from pointnet2_modules import PointnetSAModuleVotes
import pointnet2_utils

def decode_scores(net, end_points, num_heading_bin):
    # B, num_proposal, ...
    net_transposed = net.transpose(2,1)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    # B, num_proposal, 2
    objectness_scores = net_transposed[:,:,0:2]
    end_points['objectness_scores'] = objectness_scores
    
    base_xyz = end_points['aggregated_vote_xyz']  # (B, num_proposal, 3)
    center = base_xyz + net_transposed[:,:,2:5]   # (B, num_proposal, 3)
    end_points['center'] = center

    heading_scores = net_transposed[:,:,5:5+num_heading_bin*3]
    heading_residuals_normalized = net_transposed[:,:,5+num_heading_bin*3:5+num_heading_bin*3*2]
    end_points['heading_scores'] = heading_scores                                             # (B,num_proposal,num_heading_bin*3)
    end_points['heading_residuals_normalized'] = heading_residuals_normalized                 # (B,num_proposal,num_heading_bin*3) (should be -1 to 1)
    end_points['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin)  # (B,num_proposal,num_heading_bin) (should be -7.5 to 7.5)

    return end_points


class ProposalModule(nn.Module):
    def __init__(self, num_heading_bin, num_proposal, sampling, seed_feat_dim=256):
        super().__init__()

        self.num_heading_bin = num_heading_bin  # 12
        self.num_proposal = num_proposal        # 32
        self.sampling = sampling                # random
        self.seed_feat_dim = seed_feat_dim      # 256

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_proposal,
                radius=4,
                nsample=16,
                mlp=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )
    
        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2)
        self.conv1 = torch.nn.Conv1d(128,128,1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.conv3 = torch.nn.Conv1d(128,2+3+num_heading_bin*3*2,1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

    def forward(self, xyz, features, end_points):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps': 
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape[1]
            batch_size = end_points['seed_xyz'].shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            log_string('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()
        end_points['aggregated_vote_xyz'] = xyz           # (batch_size,num_proposal,3)
        end_points['aggregated_vote_inds'] = sample_inds  # (batch_size,num_proposal)

        # --------- PROPOSAL GENERATION ---------
        net = F.relu(self.bn1(self.conv1(features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net)  # (batch_size, 2+3+num_heading_bin*3*2, num_proposal)

        end_points = decode_scores(net, end_points, self.num_heading_bin)
        return end_points

if __name__=='__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
    net = ProposalModule(DC.num_heading_bin,
        128, 'seed_fps').cuda()
    end_points = {'seed_xyz': torch.rand(8,1024,3).cuda()}
    out = net(torch.rand(8,1024,3).cuda(), torch.rand(8,256,1024).cuda(), end_points)
    for key in out:
        print(key, out[key].shape)
