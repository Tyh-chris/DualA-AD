import itertools

import torch
from matplotlib import pyplot as plt
from mpmath import diffs
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from torchvision import  models


class CLLoss(_Loss):
    def __init__(self, device: str = 'gpu') -> None:
        super().__init__()
        self.device = device
        self.loss_network = models.vgg19(pretrained=True).features.to(self.device)

    #重建图之间计算相似矩阵和原图之间计算矩阵的对比
    def comput_pair_recandOri(self,recImage,oriImage):
        with torch.no_grad():
            rec_feature = self.loss_network(recImage) #512,7,7
            rec_flat=rec_feature.flatten(start_dim=1)
            ori_feature = self.loss_network(oriImage)
            ori_flat = ori_feature.flatten(start_dim=1)
        rec_flat = F.normalize(rec_flat, dim=1)
        ori_flat = F.normalize(ori_flat, dim=1)
        similarty_matrix_rec = torch.mm(rec_flat, rec_flat.t())
        similarty_matrix_ori = torch.mm(ori_flat, ori_flat.t())
        # similarty_matrix_rec=cosine_similarity(rec_flat,rec_flat)
        # similarty_matrix_ori=cosine_similarity(ori_flat,ori_flat)
        dif_similatity=torch.mean((similarty_matrix_ori-similarty_matrix_rec)**2)

        #可视化相似度矩阵
        # plt.figure(figsize=(10,8))
        # sns.heatmap(similarty_matrix_rec,annot=True,cmap='coolwarm',fmt=".2f")
        return dif_similatity


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x1=torch.rand(1,3, 224, 224).to(device=device)
    x2=torch.rand(1,3, 224, 224).to(device=device)
    c=CLLoss(device=device)
    d=c.comput_pair_recandOri(x1,x2)
