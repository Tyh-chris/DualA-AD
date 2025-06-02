# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
import network.util.misc as misc
from network.util.misc import NativeScalerWithGradNormCount as NativeScaler
import network.models_mae_fft as models_mae
from engine_pretrain_twoD import train_one_epoch, unpatchify
from network.discriminator import SimpleDiscriminator
from alert import GanAlert
from dataset_chest import data_test

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args, type):

    misc.init_distributed_mode(args)
    image_path = os.path.join('/home/cquml/tyh/data/chest', type)
    args.output_dir = os.path.join(args.output_dir, 'train-1')
    args.log_dir = os.path.join(args.log_dir, 'train-1')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    test_save = os.path.join(args.output_dir, 'test_save')
    test_save_best0 = os.path.join(test_save, 'best')
    if not os.path.exists(test_save_best0):
        os.makedirs(test_save_best0)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train_1 = transforms.Compose([
        transforms.Resize([224, 224]),  # 3 is bicubicp
        transforms.RandomAffine(10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_test_1 = transforms.Compose([
        transforms.Resize([224, 224]),  # 3 is bicubic
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(os.path.join(image_path, 'train'), transform=transform_train_1)
    dataset_test = data_test(os.path.join(image_path, 'test'), transform=transform_test_1)
    print(print("  >>> Total # of train(ben) sampler : %d" % (len(dataset_train))))
    print(print("  >>> Total # of test(ben) sampler : %d" % (len(dataset_test))))
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.to(device)

    # 加载vit imagenet权重
    pretrained_weights = torch.load('../network/model_pth/mae_pretrain_vit_base_imagenet.pth')['model']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # 加载鉴别器
    discriminator = SimpleDiscriminator(size=7).cuda()
    opt_d = torch.optim.AdamW(discriminator.parameters(), betas=(0.5, 0.999), lr=1e-4)

    discriminator_fft = SimpleDiscriminator(size=7,in_cha=1).cuda()
    opt_d_fft = torch.optim.AdamW(discriminator_fft.parameters(), betas=(0.5, 0.999), lr=1e-4)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_auc_0_test = 0
    itss = []
    timeings=np.zeros((args.epochs,1))

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats,its = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            opt_d, discriminator,discriminator_fft,opt_d_fft,
            log_writer=log_writer,
            args=args,timeings=timeings
        )
        itss.append(its)
        if args.output_dir and (epoch % 50 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, type=type)
            torch.save(discriminator.state_dict(), os.path.join(args.output_dir, ('discriminator_%s.pth' % epoch)))
        log_stats = { 'epoch': epoch, **{f'train_{k}': v for k, v in train_stats.items()},
                      }
        # 验证
        transform_test = transforms.Compose([
            transforms.Resize([224, 224]),  # 3 is bicubic
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        datapath = os.path.join('/home/cquml/tyh/data/chest', type)
        dataset_train_val = datasets.ImageFolder(os.path.join(datapath, 'train'), transform=transform_test)

        alert = GanAlert(discriminator=discriminator, device=device, train_dataset=dataset_train_val, inchnal=3,
                         generator=model)

        # test mask=0
        scores, labels, val_loss = val(data_loader_test, model, discriminator, optimizer, mask_ratio=0)
        results_0_test = alert.evaluate(scores, labels, collect=True, train=True, mask_ratio=0)
        imgauc_0_test = results_0_test['auc']
        with open(os.path.join(test_save_mask0, "test_log.txt"), mode="a", encoding="utf-8") as f:
            f.write('【test】epoch:' + str(epoch))
            f.write(str(results_0_test) + "\n")
        if best_auc_0_test < imgauc_0_test:
            with open(os.path.join(test_save_mask0, "test_log.txt"), mode="a", encoding="utf-8") as f:
                f.write('best' + "\n")
            print('[test]mask_0:', results_0_test)
            best_auc_0_test = imgauc_0_test
            torch.save(discriminator.state_dict(),
                       os.path.join(test_save_best0, 'discriminator_best.pth'))
            checkpoint_paths = [
                os.path.join(test_save_best0, 'generator_vit_base_best.pth')]
            for checkpoint_path in checkpoint_paths:
                to_save = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }
            torch.save(to_save, checkpoint_path)

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    itss_mean = sum(itss) / len(itss)
    print("it/s:" + str(itss_mean))
    mean_syn = np.sum(timeings) / args.epochs
    mean_fps = 1000. / mean_syn
    print('mean_fps:' + str(mean_fps))

    log_writer.close()


def val(dataloader, model, discriminator, opt, mask_ratio):
    model.eval()
    discriminator.eval()
    tot_loss = {'recon_l1': 0.}
    # for anomaly score
    scores = []
    # for gt labels
    labels = []

    count = 0
    for i, (img, label, filenames) in enumerate(dataloader):
        count += img.shape[0]
        img = img.to(device)
        label = label.to(device)
        opt.zero_grad()
        loss, pred, mask = model(img, mask_ratio)  # b,N,dim
        rec_img = unpatchify(pred,3)  # b,c,w,h
        fake_v = discriminator(rec_img)
        scores += list(fake_v.detach().cpu().numpy())
        labels += list(label.detach().cpu().numpy())

    return scores, labels, tot_loss


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    type = 'ZhangLabData'
    # type = 'chexpert'
    # type = 'RSNA'
    # type = 'VinCXR'
    # type = 'OCT'
    args.output_dir = os.path.join(args.output_dir, type)
    args.log_dir = os.path.join(args.log_dir, type)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, type)
