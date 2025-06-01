# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
import time
from typing import Iterable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import mae.util.misc as misc
import mae.util.lr_sched as lr_sched
from mae.optim.cl_loss import CLLoss
from mae.optim.image_losses import PerceptualLoss
from focal_frequency_loss import FocalFrequencyLoss as FFL

def train_one_epoch(
        model: torch.nn.Module,
        data_loader: Iterable, optimizer: torch.optim.Optimizer,
        device, epoch: int, loss_scaler,
        opt_d: torch.optim.Optimizer,
        discriminator: torch.nn.Module,
        discriminator_fft:torch.nn.Module,
        opt_d_fft:torch.optim.Optimizer,
        log_writer=None,
        args=None,timeings=None
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    starter_time = time.time()
    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        # 鉴别器的训练和损失
        loss_d, loss, rec_nopatch, mask = train_discriminator(samples, model, opt_d, discriminator)
        loss_rec=loss.item()

        #频域判别器损失
        d_loss_fft=train_discriminator_fft(samples,rec_nopatch.to(torch.float32),opt_d_fft,discriminator_fft)
        loss+=d_loss_fft

        # gan损失  生成网络为了让判别网络将自己生成的样本判别为真实样本。
        fake_validity = discriminator(rec_nopatch.to(torch.float32))
        ce = nn.BCEWithLogitsLoss().cuda()
        g_loss = 0.005 * ce(fake_validity, torch.ones_like(fake_validity))
        loss += g_loss

        # 频域gan损失  生成网络为了让判别网络将自己生成的样本判别为真实样本。
        x_freq_rec=transToFFt(rec_nopatch.to(torch.float32)).detach().to(torch.float32)
        fake_validity_fft = discriminator_fft(x_freq_rec)
        ce = nn.BCEWithLogitsLoss().cuda()
        g_loss_fft = 0.005 * ce(fake_validity_fft, torch.ones_like(fake_validity_fft))
        loss += g_loss_fft


        criterion_PL = PerceptualLoss(device=device)
        alfa =0.05
        loss_pl = alfa * criterion_PL(rec_nopatch.to(torch.float32), samples)
        loss+=loss_pl
        #
        CL_loss=CLLoss(device=device)
        loss_pair=CL_loss.comput_pair_recandOri(rec_nopatch.to(torch.float32),samples)
        loss+=loss_pair

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        end_time = time.time()
        torch.cuda.synchronize()
        cur_time = end_time - starter_time
        timeings[epoch] = cur_time

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_d=loss_d)
        metric_logger.update(loss_d_fft=d_loss_fft)
        metric_logger.update(loss_gan=g_loss.item())
        metric_logger.update(loss_rec=loss_rec)
        metric_logger.update(loss_gsn_fft=g_loss_fft)
        metric_logger.update(loss_pair=loss_pair)
        metric_logger.update(loss_pl=loss_pl.item())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_d', loss_d, epoch_1000x)
            log_writer.add_scalar('train_loss_gan', g_loss.item(), epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},metric_logger.its


def train_discriminator(img, model, opt_d, discriminator):
    # img,rec_img:64,3,224,224
    ce = nn.BCEWithLogitsLoss().cuda()
    opt_d.zero_grad()
    with torch.cuda.amp.autocast():
        loss, pred, mask = model(img)  # b,N,dim

    #使用restruction的特征
    rec_img = unpatchify(pred, 3)  # b,c,w,h
    real_validity = discriminator(img)  # 16，1
    # Fake images
    fake_validity = discriminator(rec_img.detach().to(torch.float32))

    # cross_entropy loss
    d_loss = ce(real_validity, torch.ones_like(real_validity))
    d_loss += ce(fake_validity, torch.zeros_like(fake_validity))
    d_loss *= 1.

    d_loss.backward()
    opt_d.step()

    return d_loss.item(), loss, rec_img, mask


def unpatchify(rec, inchanl):
    p = 16
    h = w = int(rec.shape[1] ** .5)
    assert h * w == rec.shape[1]
    rec = rec.reshape(shape=(rec.shape[0], h, w, p, p, inchanl))
    rec = torch.einsum('nhwpqc->nchpwq', rec)
    imgs = rec.reshape(shape=(rec.shape[0], inchanl, h * p, h * p))
    return imgs


def train_discriminator_fft(samples,recImgs, opt_d_fft, discriminator_fft):
    #转成一通道
    x_freq_ori=transToFFt(samples).detach().to(torch.float32)
    x_freq_rec=transToFFt(recImgs).detach().to(torch.float32)

    ce = nn.BCEWithLogitsLoss().cuda()
    real_validity = discriminator_fft(x_freq_ori)  # 16，1
    # Fake images
    fake_validity = discriminator_fft(x_freq_rec.detach().to(torch.float32))

    d_loss_fft = ce(real_validity, torch.ones_like(real_validity))
    d_loss_fft += ce(fake_validity, torch.zeros_like(fake_validity))
    d_loss_fft *= 1.

    d_loss_fft.backward()
    opt_d_fft.step()
    return d_loss_fft.item()

def transToFFt(imgs):
    grayscale_transform = transforms.Grayscale(1)
    signal_imgs = grayscale_transform(imgs)
    x_freq = torch.fft.fft2(signal_imgs)
    # shift low frequency to the center
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
    return  x_freq





