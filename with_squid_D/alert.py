import torch
import torch.nn.functional as F
import torch
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
# from matplotlib import pyplot as plt
import random
from scipy.special import expit
from sklearn.metrics import auc, roc_curve, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, \
    average_precision_score

from with_squid_D.EvalIndicators import printScore
from with_squid_D.engine_pretrain_D import unpatchify


class GanAlert(object):
    def __init__(self, discriminator, device, train_dataset,inchnal,generator=None):
        # self.args = args
        self.scores = []
        self.labels = []
        self.inchanl=inchnal
        # for vis
        self.imgs = []
        self.targets = []

        self.discriminator = discriminator
        self.generator = generator

        # self.CONFIG = CONFIG

        # self.early_stop = CONFIG.early_stop if CONFIG is not None else 200
        self.early_stop = 1000

        # training set with batch size 1
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        self.device=device

# 获取训练数据在鉴别器的均值和方差
    def collect(self, dataloader,mask_ratio):
        assert self.generator is not None
        self.generator.eval()

        all_disc = []

        for i, (img, label) in enumerate(dataloader):
            B = img.shape[0]

            img = img.to(self.device)
            label = label.to(self.device)
            loss, pred, mask= self.generator(img,mask_ratio=mask_ratio)  # b,N,dim
            rec_img = unpatchify(pred,3)
            if(self.inchanl==1):
                grayscale_transform = transforms.Grayscale(1)
                rec_img = grayscale_transform(rec_img)
            # out = self.generator(img)
            # out['recon'】是重建的图像
            fake_v = self.discriminator(rec_img)
            disc = list(fake_v.detach().cpu().numpy())
            all_disc += disc

            if i >= self.early_stop:
                break

        # calculate stats
        return np.mean(all_disc), np.std(all_disc)

    def evaluate(self, scores, labels,collect=True,train=False,mask_ratio=0.75):

        # calculate mean/std on training set?
        if collect:
            mean, std = self.collect(self.train_loader,mask_ratio)
        else:
            mean = 0.
            std = 1.

        if train:
            results = self.val_alert(scores, labels, mean, std)
            return results

        results,prediction,scores =self.alert_1(scores, labels, mean, std, print_info=False)
        return results,prediction,scores
    #
    def val_alert(self, scores, labels, mean=0., std=1., ):
        scores = np.array(scores)
        labels = np.array(labels)

        scores = (scores - mean) / std  # 真图分数大，在鉴别器中真图为1，重构图为0.
        #
        scores = 1. - expit(scores)
        # scores=1-scores  #abnormal is 1

        fpr, tpr, thres = roc_curve(labels, scores)

        auc_score = auc(fpr, tpr) * 100.
        maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
        threshold_roc = thres[maxindex]
        prediction = np.zeros_like(scores)
        prediction[scores >= threshold_roc] = 1
        prediction = prediction.ravel()
        scores = scores.ravel()

        # metrics
        f1 = f1_score(labels, prediction) * 100.
        acc = np.average(prediction == labels) * 100.
        recall = recall_score(labels, prediction) * 100.
        precision = precision_score(labels, prediction, labels=np.unique(prediction)) * 100.
        tn, fp, fn, tp = confusion_matrix(labels, prediction).ravel()
        specificity = (tn / (tn + fp)) * 100.

        results = dict(threshold=threshold_roc, auc=auc_score, acc=acc, f1=f1, recall=recall, precision=precision,
                       specificity=specificity)
        # print('val_result', results)
        return results

    def alert(self, scores, labels, mean=0., std=1., print_info=True):
        scores = np.array(scores)
        labels = np.array(labels)
        
        scores = (scores - mean) / std   #真图分数大，在鉴别器中真图为1，重构图为0.

        scores = 1. - expit(scores) # 1 is anomaly!!   expit：logistic sigmoid函数映射到【0，1】概率分布的有效实数空间   正常图得到的score大，expit(score)大，但0是normal
 
        best_acc = -1
        best_t = 0

        fpr, tpr, thres = roc_curve(labels, scores)
        # np.save('fpr_CGDualMAE-83.59.npy', fpr, )
        # np.save('tpr_CGDualMAE-83.59.npy', tpr)
        auc_score = auc(fpr, tpr) * 100.

        for t in thres:
            prediction = np.zeros_like(scores)
            prediction[scores >= t] = 1
            prediction=prediction.ravel()
            scores=scores.ravel()

            # metrics
            f1 = f1_score(labels, prediction) * 100.
            acc = np.average(prediction == labels) * 100.
            recall=recall_score(labels, prediction) * 100.  #灵敏度，召回率
            precision = precision_score(labels, prediction, labels=np.unique(prediction)) * 100.
            tn, fp, fn, tp = confusion_matrix(labels, prediction).ravel()
            specificity = (tn / (tn+fp)) * 100.   # 特异性

            if acc > best_acc:
                best_t = t
                best_acc = acc
                results = dict(threshold=t, auc=auc_score, acc=best_acc, f1=f1, recall=recall, precision=precision, specificity=specificity)

            if print_info:
                print('threshold: %.2f, auc: %.2f, acc: %.2f, f1: %.2f, recall(sens): %.2f, prec: %.2f, spec: %.2f' % (t, auc_score, acc, f1, recall, precision, specificity))

        if print_info:
            print('[BEST] threshold: %.2f, auc: %.2f, acc: %.2f, f1: %.2f, recall(sens): %.2f, prec: %.2f, spec: %.2f' % (results['threshold'], results['auc'], results['acc'], results['f1'], results['recall'], results['precision'], results['specificity']))
        print(results)
        return results

    def alert_1(self, scores, labels, mean=0., std=1.,  print_info=True):
        scores = np.array(scores)  # 异常得到的负数小
        labels = np.array(labels)
        # 真图分数大，在鉴别器中真图为1，重构图为0.
        # （1）引入train data 禁锢的更高哦
        #这行是否采用train data似乎用处不大。但这样采用能够体现与正常分布的距离
        scores = (scores - mean) / std
        # 1 is anomaly!!   expit：logistic sigmoid函数映射到【0，1】概率分布的有效实数空间   正常图得到的score大，expit(score)大，但0是normal
        # 其实可以在discriminator末尾加上这个激活函数，然后在这里使用1-score。根据对抗训练真实为1，重构为0，因此需要在这里使用1-score
        scores = 1. - expit(scores)

        # （2）score 归一化【0，1】
        # scores=1-((scores - np.min(scores)) / (np.max(scores) - np.min(scores)))

        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        ap=average_precision_score(labels,scores)
        print(roc_auc)
        print(ap)
        np.save('fpr_Dual--83.59.npy', fpr, )
        np.save('tpr_CGDualMAE-83.59.npy', tpr)
        maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
        threshold_roc = thresholds[maxindex]
        # prediction = np.zeros_like(scores)
        prediction = np.where(scores < threshold_roc, 0, 1)
        prediction = prediction.ravel()
        print('threshold_roc:'+str(threshold_roc))
        print('AUC:'+str(roc_auc))
        row=printScore(prediction,labels)
        row.append(round(roc_auc,4))
        row.append(round(threshold_roc,4))
        return row,prediction,scores


import torch
import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
# from matplotlib import pyplot as plt
import random
from scipy.special import expit
from sklearn.metrics import auc, roc_curve, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, \
    average_precision_score

from with_squid_D.EvalIndicators import printScore
from with_squid_D.engine_pretrain_D import unpatchify


class GanAlert(object):
    def __init__(self, discriminator, device, train_dataset,inchnal,generator=None):
        # self.args = args
        self.scores = []
        self.labels = []
        self.inchanl=inchnal
        # for vis
        self.imgs = []
        self.targets = []

        self.discriminator = discriminator
        self.generator = generator

        # self.CONFIG = CONFIG

        # self.early_stop = CONFIG.early_stop if CONFIG is not None else 200
        self.early_stop = 1000

        # training set with batch size 1
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        self.device=device

# 获取训练数据在鉴别器的均值和方差
    def collect(self, dataloader,mask_ratio):
        assert self.generator is not None
        self.generator.eval()

        all_disc = []

        for i, (img, label) in enumerate(dataloader):
            B = img.shape[0]

            img = img.to(self.device)
            label = label.to(self.device)
            loss, pred, mask= self.generator(img,mask_ratio=mask_ratio)  # b,N,dim
            rec_img = unpatchify(pred,3)
            if(self.inchanl==1):
                grayscale_transform = transforms.Grayscale(1)
                rec_img = grayscale_transform(rec_img)
            # out = self.generator(img)
            # out['recon'】是重建的图像
            fake_v = self.discriminator(rec_img)
            disc = list(fake_v.detach().cpu().numpy())
            all_disc += disc

            if i >= self.early_stop:
                break

        # calculate stats
        return np.mean(all_disc), np.std(all_disc)

    def evaluate(self, scores, labels,collect=True,train=False,mask_ratio=0.75):

        # calculate mean/std on training set?
        if collect:
            mean, std = self.collect(self.train_loader,mask_ratio)
        else:
            mean = 0.
            std = 1.

        if train:
            results = self.val_alert(scores, labels, mean, std)
            return results

        results,prediction,scores =self.alert(scores, labels, mean, std, print_info=False)
        return results,prediction,scores
    #
    def val_alert(self, scores, labels, mean=0., std=1., ):
        scores = np.array(scores)
        labels = np.array(labels)

        scores = (scores - mean) / std  # 真图分数大，在鉴别器中真图为1，重构图为0.
        #
        scores = 1. - expit(scores)
        # scores=1-scores  #abnormal is 1

        fpr, tpr, thres = roc_curve(labels, scores)

        auc_score = auc(fpr, tpr) * 100.
        ap = average_precision_score(labels, scores)
        maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
        threshold_roc = thres[maxindex]
        prediction = np.zeros_like(scores)
        prediction[scores >= threshold_roc] = 1
        prediction = prediction.ravel()
        scores = scores.ravel()

        # metrics
        f1 = f1_score(labels, prediction) * 100.
        acc = np.average(prediction == labels) * 100.
        recall = recall_score(labels, prediction) * 100.
        precision = precision_score(labels, prediction, labels=np.unique(prediction)) * 100.
        tn, fp, fn, tp = confusion_matrix(labels, prediction).ravel()
        specificity = (tn / (tn + fp)) * 100.

        results = dict(threshold=threshold_roc, auc=auc_score, acc=acc, f1=f1, ap=ap,recall=recall, precision=precision,
                       specificity=specificity)
        # print('val_result', results)
        return results

    def alert(self, scores, labels, mean=0., std=1.,  print_info=True):
        scores = np.array(scores)  # 异常得到的负数小
        labels = np.array(labels)
        # 真图分数大，在鉴别器中真图为1，重构图为0.
        # （1）引入train data 禁锢的更高哦
        #这行是否采用train data似乎用处不大。但这样采用能够体现与正常分布的距离
        scores = (scores - mean) / std
        # 1 is anomaly!!   expit：logistic sigmoid函数映射到【0，1】概率分布的有效实数空间   正常图得到的score大，expit(score)大，但0是normal
        # 其实可以在discriminator末尾加上这个激活函数，然后在这里使用1-score。根据对抗训练真实为1，重构为0，因此需要在这里使用1-score
        scores = 1. - expit(scores)

        # （2）score 归一化【0，1】
        # scores=1-((scores - np.min(scores)) / (np.max(scores) - np.min(scores)))

        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        ap=average_precision_score(labels,scores)
        # print(roc_auc)
        print('AP:'+str(ap))
        # np.save('fpr_DualMAE-%.2f.npy'% roc_auc, fpr)
        # np.save('tpr_DualMAE-%.2f.npy' % roc_auc, tpr)
        maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
        threshold_roc = thresholds[maxindex]
        # prediction = np.zeros_like(scores)
        prediction = np.where(scores < threshold_roc, 0, 1)
        prediction = prediction.ravel()
        print('threshold_roc:'+str(threshold_roc))
        print('AUC:'+str(roc_auc))
        row=printScore(prediction,labels)
        row.append(round(roc_auc,4))
        row.append(round(threshold_roc,4))
        return row,prediction,scores


