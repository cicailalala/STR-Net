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
from dataloaders import tn3k, ddti, tnus
from dataloaders import utils
# Custom includes
from visualization.metrics import Metrics, evaluate, evaluate_iou
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
    parser.add_argument('-data_root', type=str, default='./data/')
    parser.add_argument('-save_path', type=str, default='./models/')
    parser.add_argument('-test_dataset', type=str, default='TATN_TN3K')
    parser.add_argument('-test_fold', type=str, default='test')
    parser.add_argument('-fold', type=int, default=0)
    return parser.parse_args()

def get_edge(img):
    img[img>=128] = 255
    img[img<128] = 0
    h, w = img.shape
    h_s, h_e, w_s, w_e = -1, -1, -1, -1

    
    if np.sum(img[0,:]) > 0:
        h_s = 0
    if np.sum(img[h-1,:]) > 0:
        h_e = h
    if np.sum(img[:,0]) > 0:
        w_s = 0
    if np.sum(img[:,w-1]) > 0:
        w_e = w
    
    for i in range(h-1):
        if np.sum(img[i,:])==0 and np.sum(img[i+1,:])>0 and h_s==-1:
            h_s = i+1
        if np.sum(img[i,:])>0 and np.sum(img[i+1,:])==0 and h_e==-1:
            h_e = i+1
    for j in range(w-1):
        if np.sum(img[:,j])==0 and np.sum(img[:,j+1])>0 and w_s==-1:
            w_s = j+1
        if np.sum(img[:,j])>0 and np.sum(img[:,j+1])==0 and w_e==-1:
            w_e = j+1 
                      
    if h_s - 5 < 0:
        h_s = 0
    else:
        h_s = h_s - 5
        
    if h_e + 5 > h:
        h_e = h
    else:
        h_e = h_e + 5
        
    if w_s - 5 < 0:
        w_s = 0
    else:
        w_s = w_s - 5
        
    if w_e + 5 > w:
        w_e = w
    else:
        w_e = w_e + 5       
    return h_s, h_e, w_s, w_e

def check_ls(pred_name, files, test_dataset, Processd_n):
    if test_dataset == 'TATN_TN3K':
        ls_name = pred_name[:4] + 'l_'
    elif test_dataset == 'TATN_DDTI' or test_dataset == 'TATU_DDTI':
        ls_name = pred_name.split('s')[0] + 'l_'
    elif test_dataset == 'TATU_TNUS':
        ls_name = pred_name.split('s')[0] + 'l_'           
    elif test_dataset == 'TATN_SC2KT' or test_dataset == 'TATN_SC2KT':
        ls_name = pred_name.split('s')[0] + 'l_'
    elif test_dataset == 'TATN_S2KT' or test_dataset == 'TATN_S2KT':
        ls_name = pred_name.split('s')[0] + 'l_'
        
    for _name in files:
        if ls_name in _name:
            if _name.split('l')[0] == pred_name.split('s')[0]:
                if '0.0.' in pred_name:
                    print('Processd!!!!!!!!!!', Processd_n, pred_name, _name)
                    Processd_n += 1
                    return False, Processd_n 
                else:
                    print('Processd!', Processd_n, pred_name, _name)
                    Processd_n += 1
                    return False, Processd_n                         
    return True, Processd_n 

def main(args):
    args.save_dir = args.save_dir + 'results/'
    save_dir = args.save_dir + os.sep + args.test_fold + '-' + args.test_dataset + os.sep + args.model_name + os.sep + 'fold' + str(args.fold) + os.sep
    files = sorted(os.listdir(save_dir))
    print(save_dir)
    print(len(files))
    metrics = Metrics(['precision', 'recall', 'specificity', 'F1_score', 'auc', 'acc', 'iou', 'dice', 'mae', 'hd'])
    Processd_n = 0
    for pred_name in files:
        if 's_' in pred_name:
            NoProcessed, Processd_n = check_ls(pred_name, files, args.test_dataset, Processd_n)
            if NoProcessed:            
                
                pred_path = os.path.join(save_dir, pred_name)
                print('Processing ', pred_path)
                if args.test_dataset == 'TATN_TN3K':
                    gt_name = pred_name[:4] + '.jpg'
                    gt_path = os.path.join(args.data_root + 'tn3k/test-mask/', gt_name)
                elif args.test_dataset == 'TATN_DDTI' or args.test_dataset == 'TATU_DDTI':
                    gt_name = pred_name.split('s')[0] + '.PNG'
                    gt_path = os.path.join(args.data_root + 'DDTI/2_preprocessed_data/stage1/p_mask/', gt_name)
                elif args.test_dataset == 'TATU_TNUS':
                    gt_name = pred_name.split('s')[0] + '.png'
                    gt_path = os.path.join(args.data_root + 'tnus/part2_for_seg/test-mask/', gt_name)
                elif args.test_dataset == 'TATN_SC2KT' or args.test_dataset == 'TATU_SC2KT':
                    gt_name = pred_name.split('s')[0] + '.png'
                    gt_path = os.path.join(args.data_root + 'sc2kt/test-mask/', gt_name)
                elif args.test_dataset == 'TATN_S2KT' or args.test_dataset == 'TATU_S2KT':
                    gt_name = pred_name.split('s')[0] + '.png'
                    gt_path = os.path.join(args.data_root + 's2kt/test-mask/', gt_name)                


                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                h, w = pred.shape
                
                if np.max(pred) != 255:
                    print(pred_name, np.max(pred), np.sum(pred))
                    save_name = pred_name.replace('s', 'l')
                    save_path = os.path.join(save_dir, save_name)
                    cv2.imwrite(save_path, pred)
                else:
                
                    initial_lsf = np.zeros(pred.shape)
                    initial_lsf[pred<1] = 2.0
                    initial_lsf[pred>=1] = -2.0
                    
                
                    # parameters
                    params = {
                        'img': pred,
                        'initial_lsf': initial_lsf,
                        'timestep': 5,  # time step
                        'iter_inner': 5,
                        'iter_outer': 40,
                        'lmda': 5,  # coefficient of the weighted length term L(phi)
                        'alfa': 1.5,  # coefficient of the weighted area term A(phi)
                        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
                        'sigma': 1.5,  # scale parameter in Gaussian kernel
                        'potential_function': DOUBLE_WELL,
                    }     
                    
                    phi = find_lsf(**params)  
                    # draw_all(phi, params['img'], 10)  
                    
                    print(phi.shape, np.unique(phi, return_counts=True))
                    print(phi)
                    # pred_ls = phi[5:h+5, 5:w+5]
                    pred_ls = phi
                    pred_ls[pred_ls>=0]=0
                    pred_ls[pred_ls<0]=1
        
                    pred = pred/255
                    gt = gt/255
                    pred[pred>=0.5] = 1
                    pred[pred<0.5] = 0
                    gt[gt>=0.5] = 1
                    gt[gt<0.5] = 0
                    print(np.unique(pred, return_counts=True), np.unique(pred_ls, return_counts=True), np.unique(gt, return_counts=True))
                    
                                
                    pred_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
                    pred_ls_tensor = torch.from_numpy(pred_ls).unsqueeze(0).unsqueeze(0)
                    gt_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
                    _iou = evaluate_iou(pred_tensor, gt_tensor)
                    print(pred_path, pred.shape, _iou.data.numpy())
                    _iou = evaluate_iou(pred_ls_tensor, gt_tensor)
                    metrics.update(iou=_iou)
                    
        
                    save_png = pred_ls
                    save_png = save_png * 255
                    save_png = save_png.astype(np.uint8)
                    
                    if args.test_dataset == 'TATN_TN3K':
                        save_name = pred_name[:4] + 'l_' + str(_iou.data.numpy())[:6] + '.jpg'
                    elif args.test_dataset == 'TATU_TNUS':
                        save_name = pred_name.split('s')[0] + 'l_' + str(_iou.cpu().data.numpy())[:6] + '.png'
                        
                    elif args.test_dataset == 'TATN_DDTI' or args.test_dataset == 'TATU_DDTI':
                        save_name = pred_name.split('s')[0] + 'l_' + str(_iou.data.numpy())[:6] + '.PNG'
                    elif args.test_dataset == 'TATN_SC2KT' or args.test_dataset == 'TATU_SC2KT':
                        save_name = pred_name.split('s')[0] + 'l_' + str(_iou.data.numpy())[:6] + '.png'
                    elif args.test_dataset == 'TATN_S2KT' or args.test_dataset == 'TATU_S2KT':
                        save_name = pred_name.split('s')[0] + 'l_' + str(_iou.cpu().data.numpy())[:6] + '.png'                    
                    
                    
                    save_path = os.path.join(save_dir, save_name)
                    cv2.imwrite(save_path, save_png)   
                    print(save_path, pred.shape, _iou.data.numpy())     
            
    metrics_result = metrics.mean(len(files)/2)
    print("Test Result:")
    print('recall: %.4f, specificity: %.4f, precision: %.4f, F1_score:%.4f, acc: %.4f, iou: %.4f, mae: %.4f, dice: %.4f, hd: %.4f, auc: %.4f'
        % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
           metrics_result['F1_score'],
           metrics_result['acc'], metrics_result['iou'], metrics_result['mae'], metrics_result['dice'],
           metrics_result['hd'], metrics_result['auc']))            
            
            
    os.exit()



if __name__ == '__main__':
    args = get_arguments()
    main(args)
