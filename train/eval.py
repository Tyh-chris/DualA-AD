import time

import numpy as np
import torch
from matplotlib import pyplot as plt
# from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets

from network import models_mae_fft
from train.dataset_chest import data_test
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, auc
from train.engine_pretrain_D import unpatchify

torch.set_printoptions(10)
import os
from network.discriminator import SimpleDiscriminator
from alert import GanAlert


device='cuda' if torch.cuda.is_available() else 'cpu'

type='ZhangLabData'
save_path='output_dir/%s/train-1'%type
model_path='output_dir/%s/train-1'%type
# model_path='output_dir/%s/vit_base'%type
if not os.path.exists(save_path):
    os.makedirs(save_path)
batch_size=1
epoch=114
mask_ratio=0
# data
transform_test=transforms.Compose([
        transforms.Resize([224,224]),  # 3 is bicubic
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


datapath=os.path.join('/home/cquml/tyh/data/chest',type)
dataset_train = datasets.ImageFolder(os.path.join(datapath,'train'), transform=transform_test)

test_data= data_test(os.path.join(datapath,'test'))
dataloader_test = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
# build main model from exp folder
model = models_mae_fft.mae_vit_base_patch16_dec512d8b(norm_pix_loss=False)
print('Loading AE...')
path=os.path.join(model_path,'generator_vit_base_best.pth')
ckpt = torch.load(path)

model.load_state_dict(ckpt['model'])
model.to(device)
print('AE loaded!'+path)

# for discriminator
discriminator = SimpleDiscriminator(size=7).cuda()
print('Loading discriminator...')

path=os.path.join(model_path,'discriminator_best.pth')
ckpt = torch.load(path)
discriminator.load_state_dict(ckpt)
discriminator.to(device)
print('discriminator loaded!'+path)

# alert
alert = GanAlert(discriminator=discriminator, device=device, train_dataset=dataset_train, inchnal=3,generator=model)

def plot_roc(labels, scores, filename, modelname="", save_plots=False):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # plot roc
    if save_plots:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic {modelname}')
        plt.legend(loc="lower right")
        plt.show()
        # plt.savefig(filename)
        # plt.close()

def evaluation():
    reconstructed, inputs, scores, labels,filenames = test(dataloader_test)
    results,prediction,scores_new = alert.evaluate(scores, labels,collect=True,mask_ratio=mask_ratio)
    plot_roc(labels, scores_new, save_path + "/roc_plot.png", modelname='model_%s_dis_%s' % (epoch, epoch),
             save_plots=True)


def test(dataloader):
    model.eval()
    discriminator.eval()
    # for reconstructed img
    reconstructed = []
    # for input img
    inputs = []
    # for anomaly score
    scores = []
    # for gt labels
    labels = []
    filenames=()
    count = 0
    latencies=[]
    for i, (img, label,filename) in enumerate(dataloader):

        count += img.shape[0]
        img = img.to(device)
        label = label.cpu()
        start_time = time.perf_counter_ns()  # 纳秒级计时
        loss, pred, mask = model(img,mask_ratio=mask_ratio)  # b,N,dim  ，这里看是否不加掩蔽
        rec_img = unpatchify(pred,3)  # b,c,w,h
        fake_v = discriminator(rec_img.detach())
        torch.cuda.synchronize()
        # ----计时结束
        end_time = time.perf_counter_ns()
        # 计算单次延迟
        latency_ms = (end_time - start_time) / 1e6
        latencies.append(latency_ms)
        scores += list(fake_v.detach().cpu().numpy())
        labels += list(label.detach().cpu().numpy())
        reconstructed += list(rec_img.detach().cpu().numpy())
        inputs += list(img.detach().cpu().numpy())
        filenames = filenames + filename
    avg_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)
    print(f"\nLatency Stats (batch_size=1):")
    print(f"Average: {avg_latency:.2f} ms | P99: {p99_latency:.2f} ms")
    return reconstructed, inputs, scores, labels,filenames


if __name__ == '__main__':
    evaluation()
