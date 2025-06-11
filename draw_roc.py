import argparse
import os
from itertools import cycle

import matplotlib.axes
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataloaders import tn3k, tg3k, tatn, ddti
from dataloaders import custom_transforms as trforms

# Model includes
from model.mtnet import MTNet
from model.trfe import TRFENet
from model.trfe1 import TRFENet1
from model.trfe2 import TRFENet2
from model.trfeplus import TRFEPLUS

from torchvision.models.segmentation.segmentation import deeplabv3_resnet50
from model.deeplab_v3_plus import Deeplabv3plus
from model.fcn import FCN8s
from model.segnet import SegNet
from model.unet import Unet
from model.transunet.vit_seg_modeling import VisionTransformer as ViT_seg
from model.transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from model.cpfnet import CPFNet
from model.sgunet import SGUNet

import cv2

def init_model(model_name, num_classes):
    if 'deeplab' in model_name:
        if 'resnet101' in model_name:
            net = Deeplabv3plus(nInputChannels=3, n_classes=num_classes, os=16,
                                backbone_type='resnet101')
        elif 'resnet50' in model_name:
            net = Deeplabv3plus(nInputChannels=3, n_classes=num_classes, os=16,
                                backbone_type='resnet50')
        elif 'resnet34' in model_name:
            net = Deeplabv3plus(nInputChannels=3, n_classes=num_classes, os=16,
                                backbone_type='resnet34')
        elif 'v3' in model_name:
            net = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=1)
        else:
            raise NotImplementedError
    elif 'unet' == model_name:
        net = Unet(in_ch=3, out_ch=1)
    elif 'trfe' in model_name:
        if model_name == 'trfe1':
            net = TRFENet1(in_ch=3, out_ch=1)
        elif model_name == 'trfe2':
            net = TRFENet2(in_ch=3, out_ch=1)
        elif model_name == 'trfe':
            net = TRFENet(in_ch=3, out_ch=1)
        elif 'trfeplus' in model_name:
            net = TRFEPLUS(in_ch=3, out_ch=1)
        batch_size = 8
    elif 'mtnet' in model_name:
        net = MTNet(in_ch=3, out_ch=1)
        batch_size = 8
    elif 'segnet' in model_name:
        net = SegNet(input_channels=3, output_channels=1)
    elif 'fcn' in model_name:
        net = FCN8s(1)
    elif 'ViT' in model_name:
        config_vit = CONFIGS_ViT_seg[model_name]
        config_vit.n_classes = num_classes
        config_vit.n_skip = 3  # ???n_skip????,R50?3,???0
        if model_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(224 / 16), int(224 / 16))
        net = ViT_seg(config_vit, img_size=224, num_classes=config_vit.n_classes)
    elif 'cpfnet' in model_name:
        net = CPFNet(num_classes)
    elif 'sgunet' == model_name:
        net = SGUNet(n_classes=num_classes)
    else:
        raise NotImplementedError
    return net


def forward(model_name, net, inputs):
    if 'trfe' in model_name or model_name == 'mtnet':
        if 'trfesw' in model_name:
            outputs, _, _ = net.forward(inputs)
        else:
            outputs, _ = net.forward(inputs)
    elif 'cpfnet' in model_name:
        main_out = net(inputs)
        outputs = main_out[:,0,:,:].view(1, 1, 224, 224)
    elif 'v3' in model_name:
        outputs = net(inputs)['out']
    else:
        outputs = net.forward(inputs)
    return outputs
    


def get_model_name(model_name: str):
    """???????,??plt?label"""
    print(model_name)
    if 'deeplab' in model_name:
        standard_model_name = 'Deeplabv3+'
    elif 'fcn' in model_name:
        standard_model_name = "FCN"
    elif 'unet' == model_name or 'unet_origin' == model_name:
        standard_model_name = "Unet"
    elif 'trfe' in model_name:
        standard_model_name = "TRFE"
        if model_name == 'trfeplus':
            standard_model_name = 'TRFE+'
    elif 'sgunet' in model_name:
        standard_model_name = "SGUNet"
    elif 'mtnet' in model_name:
        standard_model_name = "MTNet"
    elif 'segnet' in model_name:
        standard_model_name = 'SegNet'
    elif 'ViT' in model_name:
        standard_model_name = "Trans-Unet"
    elif 'Pranet' in model_name:
        standard_model_name = "PraNet"
    elif 'CPFNet' in model_name or 'cpfnet' in model_name:
        standard_model_name = "CPF-Net"
    else:
        raise NotImplementedError
    return standard_model_name


def get_pred_paths(model_name: str, dataset: str, fold: int):
    root = f'./results/test-{dataset}/{model_name}/fold{fold}'
    return [os.path.join(root, x) for x in os.listdir(root) if 's' in x]


def read_img(path, resize=None):
    img = Image.open(path).convert('L')
    if resize:
        img = img.resize(resize)
    img = np.array(img)
    return img


def get_gt_dict(dataset: str):
    gt_dict = {}
    if dataset == 'TN3K':
        data_dir = "/group/eee_scchan/qiwb/data/Thyroid/tn3k/test-mask/"
    elif dataset == 'DDTI':
        data_dir = "/group/eee_scchan/qiwb/data/Thyroid/DDTI/2_preprocessed_data/stage1/p_mask/"
    else:
        raise NotImplementedError
    img_paths = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    for idx, path in enumerate(img_paths):
        gt_dict[os.path.basename(img_paths[idx])] = read_img(path, resize=(224, 224))
    return gt_dict


def calculate_roc(model_name: str, dataset: str, save_dir: str):
    if dataset == 'TN3K':
        data_dir = "/group/eee_scchan/qiwb/data/Thyroid/tn3k/test-mask/"
    elif dataset == 'DDTI':
        data_dir = "/group/eee_scchan/qiwb/data/Thyroid/DDTI/2_preprocessed_data/stage1/p_mask/"

    y_list = []
    score_list = []
    for fold in range(5):
        
        models_name = ['UNet', 'SGUNet', 'TRFE', 'FCN', 'SegNet', 'Deeplabv3+', 'CPFNet', 'TransUNet', 'TRFE+', 'MTNet', 'MC-TNS']
        if model_name == 'UNet':
            results_path = f"/group/eee_scchan/qiwb/program/Thyroid/results/test-{dataset}/unet/fold{fold}/"
        elif model_name == 'SGUNet':
            results_path = f"/group/eee_scchan/qiwb/program/Thyroid/results/test-{dataset}/sgunet/fold{fold}/"
        elif model_name == 'TRFE':
            results_path = f"/group/eee_scchan/qiwb/program/Thyroid/results/test-{dataset}/trfe/fold{fold}/"
        elif model_name == 'FCN':
            results_path = f"/group/eee_scchan/qiwb/program/Thyroid/results/test-{dataset}/fcn/fold{fold}/"
        elif model_name == 'SegNet':
            results_path = f"/group/eee_scchan/qiwb/program/Thyroid/results/test-{dataset}/segnet/fold{fold}/"                    
        elif model_name == 'Deeplabv3+':
            results_path = f"/group/eee_scchan/qiwb/program/Thyroid/results/test-{dataset}/deeplab-resnet50/fold{fold}/"
        elif model_name == 'CPFNet':
            results_path = f"/group/eee_scchan/qiwb/program/Thyroid/results/test-{dataset}/cpfnet/fold{fold}/"
        elif model_name == 'TransUNet':
            results_path = f"/group/eee_scchan/qiwb/program/Thyroid/results/test-{dataset}/R50-ViT-B_16/fold{fold}/"
        elif model_name == 'TRFE+':
            results_path = f"/group/eee_scchan/qiwb/program/Thyroid/results/test-{dataset}/trfeplus-v40/fold{fold}/"                    
        elif model_name == 'MTNet':
            results_path = f"/group/eee_scchan/qiwb/program/Thyroid/results/test-{dataset}/mtnet/fold{fold}/"
        elif model_name == 'MC-TNS':
            results_path = f"/group/eee_scchan/qiwb/program/Thyroid/results/test-{dataset}/trfeplus-v37/fold{fold}/"            
                    
        
        
        files = sorted(os.listdir(results_path))
        p_files = []
        for img_name in files:
            if model_name == 'MC-TNS':
                if 'l_' in img_name:
                    p_files.append(img_name)                
            else:
                if 'p.' in img_name:
                    p_files.append(img_name)
                    
        p_files = sorted(p_files)
        print(dataset, model_name, fold, len(p_files))
        
        
        for i in range(len(p_files)):
            p_name = p_files[i]
            
            p_path = os.path.join(results_path, p_name)
            if dataset == 'TN3K':
                gt_name = p_name[:4] + '.jpg'
                gt_path = os.path.join('/group/eee_scchan/qiwb/data/Thyroid/tn3k/test-mask/', gt_name)
            elif dataset == 'DDTI':
                if model_name == 'MC-TNS':
                    gt_name = p_name.split('l_')[0] + '.PNG'
                else:
                    gt_name = p_name.split('p')[0] + '.PNG'
                gt_path = os.path.join('/group/eee_scchan/qiwb/data/Thyroid/DDTI/2_preprocessed_data/stage1/p_mask/', gt_name)        
    
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)/255
            p_img = cv2.imread(p_path, cv2.IMREAD_GRAYSCALE)/255    
                


            score_list.append(p_img.flatten().round())
            y_list.append(gt.flatten().round())    
    y = np.concatenate(y_list)
    score = np.concatenate(score_list)
    print(dataset, model_name, y.shape, score.shape)
    fpr, tpr, threshold = roc_curve(y, score)
    # print(y.dtype)
    # df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': threshold})
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # df.to_excel(f'{save_dir}/{model_name}.xlsx')
    # print(f'saved at :{save_dir}/{model_name}.xlsx')
    return fpr, tpr, threshold


def plot_roc(fpr, tpr, ax: matplotlib.axes.Axes, model_name, color):
    roc_auc = auc(fpr, tpr)
    print(model_name, roc_auc)
    if model_name == 'trfe':
        ax.plot(fpr, tpr, label=f'{get_model_name(model_name)}(AUC=%0.4f)' % roc_auc, color='red')
    elif model_name == 'trfesw':
        ax.plot(fpr, tpr, label=f'{get_model_name(model_name)}(AUC=%0.4f)' % roc_auc, color='lime')
    else:
        # ax.plot(fpr, tpr, label=f'{get_model_name(model_name)}(AUC=%0.4f)' % roc_auc, color=color)
        ax.plot(fpr, tpr, label=f'{model_name}(AUC=%0.4f)' % roc_auc, color=color)
    print(f"{model_name}'s roc curve has been plottd")


def draw_roc_func(dataset, from_scratch=False):
    root = "run/"
    models_name = ['UNet', 'SGUNet', 'TRFE', 'FCN', 'SegNet', 'Deeplabv3+', 'CPFNet', 'TransUNet', 'TRFE+', 'MTNet', 'MC-TNS']
    color_list = ['blue', 'orange', 'black', 'purple', 'brown', 'pink', 'gray', 'gold', 'green', 'cyan', 'lime', ]
    # models_name = ['trfesw-refine']
    # color_list = ['lime']
    # marker = ["o", "v", "^", "<", ">", "D", 'd']
    # marker = cycle(marker)
    fig, ax = plt.subplots()
    ax.set_xlim([0.0, 1.0])
    if dataset == 'TN3K':
        ax.set_ylim([0.75, 1.0])
    elif dataset == 'DDTI':
        ax.set_ylim([0.5, 1.0])
    ax.set_xlabel('1-Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title(f'{dataset}')
    for i, model_name in enumerate(models_name):
        # if os.path.exists(f'./results/ruc/{dataset}/fold{fold}/{model_name}.xlsx') and not from_scratch:
        #     df = pd.read_excel(f'./results/ruc/{dataset}/fold{fold}/{model_name}.xlsx')
        #     # fpr, tpr, threshold = df['fpr'], df['tpr'], df['threshold']
        #     fpr = df['fpr']
        #     tpr = df['tpr']
        # else:
        #     print(f'./results/ruc/fold{fold}/{model_name}.xlsx not found and calculate_roc from scratch.')
        fpr, tpr, threshold = calculate_roc(model_name, dataset, f'no use')
        # plot_roc(fpr, tpr, ax, net.__class__.__name__, next(marker))
        # print(np.array(fpr))
        # fpr = np.mean(np.array(fpr), axis=0)
        # tpr = np.mean(np.array(tpr), axis=0)
        plot_roc(fpr, tpr, ax, model_name, color_list[i])
    ax.legend(loc="lower right")
    # plt.show()
    if not os.path.exists(f'./results/ruc/{dataset}/'):
        os.makedirs(f'./results/ruc/{dataset}/')
    fig.savefig(f'./results/ruc/{dataset}/{dataset}.pdf')
    print(f'figure saved at: ./results/ruc/{dataset}/{dataset}.pdf')


if __name__ == '__main__':
    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(224, 224)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])
    # datasets=['TN3K', 'DDTI']
    datasets = {
        'TN3K': tn3k.TN3K(mode='test', transform=composed_transforms_ts, return_size=True),
        'DDTI': ddti.DDTI(transform=composed_transforms_ts, return_size=True)
    }
    for dataset in datasets.keys():
        draw_roc_func(dataset, from_scratch=True)
    # net = TRFESW(in_ch=3, out_ch=1)
    # net.load_state_dict(torch.load("/home/liguanbin/TRFE-Net-for-thyroid-nodule-segmentation-main/run/trfesw_old/fold1/trfesw_best.pth"))
    # print(net)
