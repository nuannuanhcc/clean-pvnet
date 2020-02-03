import torch.nn as nn
from lib.utils import net_utils
import torch
from lib.config import cfg, args

class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net

        self.vote_crit = torch.nn.functional.smooth_l1_loss
        self.seg_crit = nn.CrossEntropyLoss()

    def forward(self, batch):
        output = self.net(batch['inp'])

        scalar_stats = {}
        loss = 0

        if 'pose_test' in batch.keys():
            return output, loss, {}, {}

        weight = batch['mask'][:, None].float()

        if 'Coco' in cfg.train.dataset:
            vote_loss = self.vote_crit(output['vertex'] * weight, batch['vertex'] * weight, reduction='none')
            n, k, h, w = vote_loss.shape
            vote_loss = vote_loss.view(n, k // 2, 2, h, w)
            vis_vertex = batch['joints_vis'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            vote_loss = (vote_loss * vis_vertex).sum()
            vote_loss = vote_loss / weight.sum() / vis_vertex.sum()
        else:
            vote_loss = self.vote_crit(output['vertex'] * weight, batch['vertex'] * weight, reduction='sum')
            vote_loss = vote_loss / weight.sum() / batch['vertex'].size(1)

        scalar_stats.update({'vote_loss': vote_loss})
        loss += vote_loss

        mask = batch['mask'].long()
        seg_loss = self.seg_crit(output['seg'], mask)
        scalar_stats.update({'seg_loss': seg_loss})
        loss += seg_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats
