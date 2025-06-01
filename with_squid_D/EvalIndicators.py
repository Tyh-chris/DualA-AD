# @Time: 2023/1/10 16:16
# 统计预测与实际的对比结果并计算指标
import os
import shutil

import numpy as np
from PIL import Image
from matplotlib import image as mpimg

from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_curve, auc


# 计算混淆矩阵
def compute_confusion_matrix(precited,expected):
    part = precited ^ expected             # 对结果进行分类，亦或使得判断正确的为0,判断错误的为1
    pcount = np.bincount(part)             # 分类结果统计，pcount[0]为0的个数，pcount[1]为1的个数
    # print(pcount)
    # true_dec=list(pcount).count(0)
    # false_dec=list(pcount).count(1)
    tp_list = list(precited & expected)    # 将TP的计算结果转换为list
    fp_list = list(precited & ~expected)   # 将FP的计算结果转换为list
    # if filenames!=None:
    #     tn_list = list(~precited & ~expected)
    #     fn_list=list(~precited & expected)
    #     tp_index = [i for i, x in enumerate(tp_list) if x == 1]
    #     for i in tp_index:
    #         shutil.copy(filenames[0][i], '/home/tyh/data/chorom_result/1/TP')
    #     fp_index = [i for i, x in enumerate(fp_list) if x == 1]
    #     for i in fp_index:
    #         shutil.copy(filenames[0][i], '/home/tyh/data/chorom_result/1/FP')
    #     tn_index = [i for i, x in enumerate(tn_list) if x == 1]
    #     for i in tn_index:
    #         shutil.copy(filenames[0][i], '/home/tyh/data/chorom_result/1/TN')
    #     fn_index = [i for i, x in enumerate(fn_list) if x == 1]
    #     for i in fn_index:
    #         shutil.copy(filenames[0][i], '/home/tyh/data/chorom_result/1/FN')
    tp = tp_list.count(1)                  # 统计TP的个数
    fp = fp_list.count(1)                  # 统计FP的个数
    tn = pcount[0] - tp                    # 统计TN的个数
    if len(pcount)==1:
        fn=0
    else:
        fn = pcount[1] - fp                    # 统计FN的个数
    return tp, fp, tn, fn

# 计算常用指标
def compute_indexes(tp, fp, tn, fn):
    accuracy = (tp+tn) / (tp+tn+fp+fn)     # 准确率
    if tp==0:
        precision=0
        recall=0
        F1=0
    else:
        precision = tp / (tp+fp)               # 正确率
        recall = tp / (tp+fn)                  # 召回率、灵敏度，真阳性
        F1 = (2*precision*recall) / (precision+recall)    # F1
    if tn!=0:
        specificity=tn/(tn+fp)    #真阴性，特异性
    return accuracy, precision, recall, F1, specificity

def printScore(precited, expected):
    # matrix = confusion_matrix(expected, precited)
    # TN, FP, FN, TP = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    tp, fp, tn, fn = compute_confusion_matrix(np.array(precited), np.array(expected))
    accuracy = round(np.average(precited == expected),4)
    recall = round(recall_score(expected, precited),4)# 召回率、灵敏度，真阳性
    precision = round(precision_score(expected, precited, labels=np.unique(precited)),4)
    F1 = round(f1_score(expected, precited),4)
    specificity = round(tn / (tn + fp),4) #真阴性，特异性
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"FN: {fn}")
    # accuracy, precision, recall, F1, specificity = compute_indexes(tp, fp, tn, fn)

    print(f"Accuracy:  {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall(tp/tp+fn):    {recall}")
    # print(f"specificity（tn/tn+fp）:    {specificity}")
    print(f"F1:        {F1}")
    # print(tp,fp,tn,fn,recall,specificity,accuracy)
    num = len(expected)
    expected=list(expected)
    precited=list(precited)
    true_abnor = expected.count(1)   # count 要list才有
    true_nor = expected.count(0)
    pre_abnor = (precited.count(1))
    row = [ num, true_abnor,
           tn,tp,fn,fp,pre_abnor, tp,
            round(tp / pre_abnor,4), recall, accuracy, round(fp / true_nor,4), round(fn / true_abnor,4),
            recall,specificity,accuracy,F1]
    return  row

def save_file(precited, expected, filenames,image_savepath):
    outprelabels, labels = np.array(precited), np.array(expected)

    tp_list = list(outprelabels & labels)  # 将TP的计算结果转换为list
    fp_list = list(outprelabels & ~labels)
    tn_list = list((~outprelabels)+2 & (~labels)+2)
    fn_list = list(~outprelabels & labels)
    TP_file=os.path.join(image_savepath,'abnormal','TP(真阳性)')
    FP_file=os.path.join(image_savepath,'abnormal','FP(假阳性)')
    TN_file=os.path.join(image_savepath,'normal','TN(真阴性)')
    FN_file=os.path.join(image_savepath,'normal','FN(假阴性)')
    os.makedirs(TP_file, exist_ok=True)
    os.makedirs(FP_file, exist_ok=True)
    os.makedirs(TN_file, exist_ok=True)
    os.makedirs(FN_file, exist_ok=True)
    tp_index = [i for i, x in enumerate(tp_list) if x == 1]
    for i in tp_index:
        # img = Image.open(filenames[i])  # 读取图片
        # out_name = filenames[i].split('/')
        # name = out_name[-2] +'_'+ out_name[-1]
        # save_path =os.path.join( TP_file , name)
        # img.save(save_path)

        shutil.copy(filenames[i], TP_file)

    fp_index = [i for i, x in enumerate(fp_list) if x == 1]
    for i in fp_index:
        shutil.copy(filenames[i], FP_file)

    tn_index = [i for i, x in enumerate(tn_list) if x == 1]
    for i in tn_index:
        # img = Image.open(filenames[i])  # 读取图片
        # out_name = filenames[i].split('/')
        # name=out_name[-2]+out_name[-1]
        # save_path =os.path.join(TN_file , name)
        # img.save(save_path)
        shutil.copy(filenames[i], TN_file)

    fn_index = [i for i, x in enumerate(fn_list) if x == 1]
    for i in fn_index:
        shutil.copy(filenames[i], FN_file)


def save_file_cam(precited, expected, filenames,image_savepath,visualizations,scores,reconstructed):
    outprelabels, labels = np.array(precited), np.array(expected)

    tp_list = list(outprelabels & labels)  # 将TP的计算结果转换为list
    fp_list = list(outprelabels & ~labels)
    tn_list = list((~outprelabels)+2 & (~labels)+2)
    fn_list = list(~outprelabels & labels)
    TP_file=os.path.join(image_savepath,'abnormal','TP(真阳性)')
    FP_file=os.path.join(image_savepath,'abnormal','FP(假阳性)')
    TN_file=os.path.join(image_savepath,'normal','TN(真阴性)')
    FN_file=os.path.join(image_savepath,'normal','FN(假阴性)')
    os.makedirs(TP_file, exist_ok=True)
    os.makedirs(FP_file, exist_ok=True)
    os.makedirs(TN_file, exist_ok=True)
    os.makedirs(FN_file, exist_ok=True)
    tp_index = [i for i, x in enumerate(tp_list) if x == 1]
    for i in tp_index:
        # img = Image.open(filenames[i])  # 读取图片
        # out_name = filenames[i].split('/')
        # name = out_name[-2] +'_'+ out_name[-1]
        # save_path =os.path.join( TP_file , name)
        # img.save(save_path)

        shutil.copy(filenames[i], TP_file)
        imagename_cam = filenames[i].split('/')[-1].split('.')[0] + 'cam_' + str(scores[i]) + '.jpg'
        mpimg.imsave(os.path.join(TP_file,imagename_cam), visualizations[i])
        imagename_rec = filenames[i].split('/')[-1].split('.')[0] + 'rec_.jpg'
        mpimg.imsave(os.path.join(TP_file,imagename_rec), reconstructed[i])
    fp_index = [i for i, x in enumerate(fp_list) if x == 1]
    for i in fp_index:
        imagename_cam = filenames[i].split('/')[-1].split('.')[0] +'cam_'+str(scores[i])+'.jpg'
        shutil.copy(filenames[i], FP_file)
        mpimg.imsave(os.path.join(FP_file,imagename_cam), visualizations[i])
        imagename_rec = filenames[i].split('/')[-1].split('.')[0] + 'rec_.jpg'
        mpimg.imsave(os.path.join(FP_file, imagename_rec), reconstructed[i])
    tn_index = [i for i, x in enumerate(tn_list) if x == 1]
    for i in tn_index:
        # img = Image.open(filenames[i])  # 读取图片
        # out_name = filenames[i].split('/')
        # name=out_name[-2]+out_name[-1]
        # save_path =os.path.join(TN_file , name)
        # img.save(save_path)

        shutil.copy(filenames[i], TN_file)
        imagename = filenames[i].split('/')[-1].split('.')[0] + 'cam_'+str(scores[i])+'.jpg'
        mpimg.imsave(os.path.join(TN_file,imagename), visualizations[i])
        imagename_rec = filenames[i].split('/')[-1].split('.')[0] + 'rec_.jpg'
        mpimg.imsave(os.path.join(TN_file, imagename_rec), reconstructed[i])

    fn_index = [i for i, x in enumerate(fn_list) if x == 1]
    for i in fn_index:
        shutil.copy(filenames[i], FN_file)
        imagename = filenames[i].split('/')[-1].split('.')[0] + 'cam_'+str(scores[i])+'.jpg'
        mpimg.imsave(os.path.join(FN_file,imagename), visualizations[i])
        imagename_rec = filenames[i].split('/')[-1].split('.')[0] + 'rec_.jpg'
        mpimg.imsave(os.path.join(FN_file, imagename_rec), reconstructed[i])

def aucscores(labels,scores):
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
    acc = np.average(prediction == np.array(labels)) * 100.
    recall = recall_score(labels, prediction) * 100.
    precision = precision_score(labels, prediction, labels=np.unique(prediction)) * 100.
    tn, fp, fn, tp = confusion_matrix(labels, prediction).ravel()
    specificity = (tn / (tn + fp)) * 100.

    results = dict(threshold=threshold_roc, auc=auc_score, acc=acc, f1=f1, recall=recall, precision=precision,
                   specificity=specificity)
    # print('val_result', results)
    return results