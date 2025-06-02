import torch
import torch.nn as nn
# from mmengine.model import BaseModule

import torch.nn.functional as F

from network.gather import concat_all_gather



class HeadLoss(nn.Module):
    """
    Pre-training head for MAE

    """
    def __init__(self,predictor,norm_pix_loss,temperature=0.07,norm_pix=False, patch_size=16,ct_weight=1.0, rc_weight=1.0 ):
        super(HeadLoss, self).__init__()

        self.norm_pix = norm_pix
        self.patch_size = patch_size
        self.predictor = predictor
        self.t = temperature
        self.ct_weight= ct_weight
        self.rc_weight = rc_weight
        self.norm_pix_loss=norm_pix_loss
        self.criterion = nn.CrossEntropyLoss()

    def patchify(self,imgs):
        """

        :param imgs:  torch.Tensor, The shape is (N, 3, H, W)
        :return: torch.Tensor, The shape is (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h,w= imgs.shape[2] // p,imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self,x):
        """

        :param x: The shape is (N, L, patch_size**2 *3)
        :return: The shape is (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def forward(self,proj_s,proj_t):
        # target = self.patchify(x)
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6) ** .5
        #
        # loss_rec = (pred_pixel - target) ** 2
        # loss_rec = loss_rec.mean(dim=-1)
        # loss_rc = (loss_rec * mask_s).sum() / mask_s.sum()
        pred_s = self.predictor(proj_s)  # 196,1,256
        pred_s = F.normalize(pred_s.squeeze(dim=1),dim=1,p=2)   #196,256
        proj_t = F.normalize(proj_t.squeeze(dim=1),dim=1,p=2)  #196,256
        proj_t = concat_all_gather(proj_t)
        #相似度计算
        score = torch.matmul(pred_s,proj_t.transpose(1, 0).detach())

        score = score / self.t  #(16,16）矩阵

        bs = score.size(0)
        label = torch.arange(bs, dtype=torch.long).to('cuda')   # 用于指示正对的位置
            # label = (torch.arange(bs, dtype=torch.long) +
            #          bs * torch.distributed.get_rank()).cuda()

            # losses = dict()
        losses_cl = self.ct_weight * 2 * self.t * self.criterion(score, label)
            # losses_rc = self.rc_weight * loss_rc

        return losses_cl







