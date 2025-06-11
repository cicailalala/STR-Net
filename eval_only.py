import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders import custom_transforms as trforms
# Dataloaders includes
from dataloaders import tn3k, ddti
from dataloaders import utils
# Custom includes
from visualization.metrics import Metrics, evaluate
import csv

from find_lsf import find_lsf
from potential_func import *
from show_fig import draw_all

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-model_name', type=str,
                        default='trfe')  # unet, mtnet, segnet, deeplab-resnet50, fcn, trfe, trfe1, trfe2
    parser.add_argument('-num_classes', type=int, default=1)
    parser.add_argument('-input_size', type=int, default=224)
    parser.add_argument('-output_stride', type=int, default=16)
    parser.add_argument('-load_path', type=str, default='./run/run_1/trfe_best.pth')
    parser.add_argument('-data_root', type=str, default='./data/')
    parser.add_argument('-save_path', type=str, default='./models/')
    parser.add_argument('-test_dataset', type=str, default='TN3K')
    parser.add_argument('-test_fold', type=str, default='test')
    parser.add_argument('-fold', type=int, default=0)
    return parser.parse_args()





def main(args):
    args.save_dir = args.save_dir + 'results/'
    save_dir = args.save_dir + os.sep + args.test_fold + '-' + args.test_dataset + os.sep + args.model_name + os.sep + 'fold' + str(args.fold) + os.sep
    files = sorted(os.listdir(save_dir))
    print(save_dir)
    print(len(files))
    metrics = Metrics(['precision', 'recall', 'specificity', 'F1_score', 'auc', 'acc', 'iou', 'dice', 'mae', 'hd'])
    
    p_files = []
    s_files = []
    l_files = []
    for img_name in files:
        if 'p.' in img_name:
            p_files.append(img_name)
        if 's_' in img_name:
            s_files.append(img_name)
        if 'l_' in img_name:
            l_files.append(img_name)
    p_files = sorted(p_files)
    s_files = sorted(s_files)
    l_files = sorted(l_files)
    print(len(p_files), len(s_files), len(l_files))
    assert len(p_files) == len(s_files) == len(l_files)
    
    improved_n = 0
    
    for i in range(len(p_files)):
        p_name = p_files[i]
        s_name = s_files[i]
        l_name = l_files[i]
        assert p_name.split('p')[0] == l_name.split('l')[0] == s_name.split('s')[0]
        
        p_path = os.path.join(save_dir, p_name)
        s_path = os.path.join(save_dir, s_name)
        l_path = os.path.join(save_dir, l_name)
        if args.test_dataset == 'TATN_TN3K':
            gt_name = s_name[:4] + '.jpg'
            gt_path = os.path.join(args.data_root + 'tn3k/test-mask/', gt_name)
        elif args.test_dataset == 'TATN_DDTI' or args.test_dataset == 'TATU_DDTI':
            gt_name = s_name.split('s')[0] + '.PNG'
            gt_path = os.path.join(args.data_root + 'DDTI/2_preprocessed_data/stage1/p_mask/', gt_name)       
        elif args.test_dataset == 'TATU_TNUS':
            gt_name = s_name.split('s')[0] + '.png'
            gt_path = os.path.join(args.data_root + 'tnus/part2_for_seg/test-mask/', gt_name)
        elif args.test_dataset == 'TATN_SC2KT' or args.test_dataset == 'TATU_SC2KT':
            gt_name = s_name.split('s')[0] + '.png'
            gt_path = os.path.join(args.data_root + 'sc2kt/test-mask/', gt_name)
        elif args.test_dataset == 'TATN_S2KT' or args.test_dataset == 'TATU_S2KT':
            gt_name = s_name.split('s')[0] + '.png'
            gt_path = os.path.join(args.data_root + 's2kt/test-mask/', gt_name)    

        
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        p_img = cv2.imread(p_path, cv2.IMREAD_GRAYSCALE)
        l_img = cv2.imread(l_path, cv2.IMREAD_GRAYSCALE)


        p_img = p_img/255
        l_img = l_img/255
        gt = gt/255
        p_img[p_img>=0.5] = 1
        p_img[p_img<0.5] = 0
        l_img[l_img>=0.5] = 1
        l_img[l_img<0.5] = 0
        gt[gt>=0.5] = 1
        gt[gt<0.5] = 0
        # print(np.unique(gt, return_counts=True), np.unique(p_img, return_counts=True), np.unique(l_img, return_counts=True))
                                    
        p_tensor = torch.from_numpy(p_img).cuda().unsqueeze(0).unsqueeze(0)
        l_tensor = torch.from_numpy(l_img).cuda().unsqueeze(0).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt).cuda().unsqueeze(0).unsqueeze(0)
        _precision, _recall, _specificity, _f1, _auc, _acc, _iou, _dice, _mae, _hd = evaluate(p_tensor, gt_tensor)         
        p_iou = _iou.cpu().data.numpy()       

        _precision, _recall, _specificity, _f1, _auc, _acc, _iou, _dice, _mae, _hd = evaluate(l_tensor, gt_tensor)
        metrics.update(recall=_recall, specificity=_specificity, precision=_precision, F1_score=_f1, acc=_acc, iou=_iou, mae=_mae, dice=_dice, hd=_hd, auc=_auc)
        print(l_name, l_img.shape, p_iou, _iou.cpu().data.numpy())
        # if _iou.cpu().data.numpy() > (p_iou-0.00001):
        if _iou.cpu().data.numpy() > p_iou:
            improved_n += 1
               

    metrics_result = metrics.mean(len(p_files))
    print("Test Result:")
    print(len(p_files), improved_n, improved_n/len(p_files))
    print('recall: %.4f, specificity: %.4f, precision: %.4f, F1_score:%.4f, acc: %.4f, iou: %.4f, mae: %.4f, dice: %.4f, hd: %.4f, auc: %.4f'
        % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
           metrics_result['F1_score'],
           metrics_result['acc'], metrics_result['iou'], metrics_result['mae'], metrics_result['dice'],
           metrics_result['hd'], metrics_result['auc'])) 


    values_txt = ['ls_'+str(args.fold), 100 * metrics_result['auc'], 100 * metrics_result['precision'], 100 * metrics_result['acc'], 100 * metrics_result['iou'], 100 * metrics_result['dice'], metrics_result['hd'], 100 * metrics_result['recall'], 100 * metrics_result['specificity'], metrics_result['mae'], 100 * metrics_result['F1_score'], 100*improved_n/len(p_files)]
    evaluation_dir = os.path.sep.join([args.save_dir, args.test_fold + '-' + args.test_dataset + '/'])
    save_path = evaluation_dir + args.model_name + '.csv'

    f = open(save_path, 'a+')
    f_csv = csv.writer(f)
    f_csv.writerow(values_txt)
    f.close()
    print(f'metrics saved in {save_path}')
    print("------------------------------------------------------------------")               

if __name__ == '__main__':
    args = get_arguments()
    main(args)
