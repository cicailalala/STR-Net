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
from dataloaders import tn3k, ddti, tnus, sc2kt, s2kt
from dataloaders import utils
# Custom includes
from visualization.metrics import Metrics, evaluate
from model.strnet import STRNet
import csv

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-model_name', type=str, default='strnet')
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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # FIXME add other models
    if 'strnet' in args.model_name:
        net = STRNet(in_ch=3, out_ch=1)
    net.load_state_dict(torch.load(args.load_path))
    net.cuda()

    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    if args.test_dataset == 'TATN_TN3K':
        test_data = tn3k.TN3K(mode='test', transform=composed_transforms_ts, return_size=True, root=args.data_root)
        gt_root = args.data_root + 'tn3k/test-mask/'
    if args.test_dataset == 'TATU_TNUS':
        test_data = tnus.TNUS(mode='test', transform=composed_transforms_ts, return_size=True, root=args.data_root)
        gt_root = args.data_root + 'tnus/part2_for_seg/test-mask/'
        
    if args.test_dataset == 'TATN_DDTI' or args.test_dataset == 'TATU_DDTI':
        test_data = ddti.DDTI(transform=composed_transforms_ts, return_size=True, root=args.data_root)
        gt_root = args.data_root + 'DDTI/2_preprocessed_data/stage1/p_mask/'
    if args.test_dataset == 'TATN_SC2KT' or args.test_dataset == 'TATU_SC2KT':
        test_data = sc2kt.SC2KT(transform=composed_transforms_ts, return_size=True, root=args.data_root)
        gt_root = args.data_root + 'sc2kt/test-mask/'
    if args.test_dataset == 'TATN_S2KT' or args.test_dataset == 'TATU_S2KT':
        test_data = s2kt.S2KT(transform=composed_transforms_ts, return_size=True, root=args.data_root)
        gt_root = args.data_root + 's2kt/test-mask/'
                
    save_dir = args.save_dir + os.sep + args.test_fold + '-' + args.test_dataset + os.sep + args.model_name + os.sep + 'fold' + str(
        args.fold) + os.sep
    testloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    num_iter_ts = len(testloader)


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    net.cuda()
    net.eval()
    with torch.no_grad():
        all_start = time.time()
        metrics = Metrics(['precision', 'recall', 'specificity', 'F1_score', 'auc', 'acc', 'iou', 'dice', 'mae', 'hd'])
        total_iou = 0
        total_cost_time = 0
        for sample_batched in tqdm(testloader):
            inputs, labels, label_name, size = sample_batched['image'], sample_batched['label'], sample_batched.get(
                'label_name'), sample_batched['size']

            labels = labels.cuda()
            inputs = inputs.cuda()
            start = time.time()
            nodule_pred, gland_pred, _, _, _, _, _, _, _, _ = net.forward(inputs)
            cost_time = time.time() - start
            prob_pred = nodule_pred
            shape = (size[0, 0], size[0, 1])
            prob_pred = F.interpolate(prob_pred, size=shape, mode='bilinear', align_corners=True)
            
            gt_path = os.path.join(gt_root, label_name[0])
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt = gt/np.max(gt)
            gt[gt>=0.5] = 1
            gt[gt<0.5] = 0
            labels= torch.from_numpy(gt).cuda().unsqueeze(0).unsqueeze(0)
            iou = utils.get_iou(prob_pred, labels)
            # print(prob_pred.shape, labels.shape)
            _precision, _recall, _specificity, _f1, _auc, _acc, _iou, _dice, _mae, _hd = evaluate(prob_pred, labels)
            metrics.update(recall=_recall, specificity=_specificity, precision=_precision,
                           F1_score=_f1, acc=_acc, iou=_iou, mae=_mae, dice=_dice, hd=_hd, auc=_auc)

            total_iou += iou
            total_cost_time += cost_time

            save_data = prob_pred[0].cpu().data
            save_png = save_data[0].numpy()
            save_saliency = save_png * 255
            save_saliency = save_saliency.astype(np.uint8)
            save_png = np.round(save_png)
            # print(save_png.shape)
            save_png = save_png * 255
            save_png = save_png.astype(np.uint8)
            save_path = save_dir + label_name[0]
            if not os.path.exists(save_path[:save_path.rfind('/')]):
                os.makedirs(save_path[:save_path.rfind('/')])
            if args.test_dataset == 'TATN_TN3K':
                save_path_s = save_dir + label_name[0].replace('.jpg', 's_'+str(iou)[:6]+'.jpg')
                cv2.imwrite(save_path_s, save_saliency)
                cv2.imwrite(save_dir + label_name[0].replace('.jpg', 'p.jpg'), save_png)
            if args.test_dataset == 'TATU_TNUS':
                save_path_s = save_dir + label_name[0].replace('.png', 's_'+str(iou)[:6]+'.png')
                cv2.imwrite(save_path_s, save_saliency)
                cv2.imwrite(save_dir + label_name[0].replace('.png', 'p.png'), save_png)
                
            if args.test_dataset == 'TATN_DDTI' or args.test_dataset == 'TATU_DDTI':
                save_path_s = save_dir + label_name[0].replace('.PNG', 's_'+str(iou)[:6]+'.PNG')
                cv2.imwrite(save_path_s, save_saliency)
                cv2.imwrite(save_dir + label_name[0].replace('.PNG', 'p.PNG'), save_png)
            if args.test_dataset == 'TATN_SC2KT' or args.test_dataset == 'TATU_SC2KT':
                save_path_s = save_dir + label_name[0].replace('.png', 's_'+str(iou)[:6]+'.png')
                cv2.imwrite(save_path_s, save_saliency)
                cv2.imwrite(save_dir + label_name[0].replace('.png', 'p.png'), save_png)
            if args.test_dataset == 'TATN_S2KT' or args.test_dataset == 'TATU_S2KT':
                save_path_s = save_dir + label_name[0].replace('.png', 's_'+str(iou)[:6]+'.png')
                cv2.imwrite(save_path_s, save_saliency)
                cv2.imwrite(save_dir + label_name[0].replace('.png', 'p.png'), save_png)                


    print(args.model_name)
    metrics_result = metrics.mean(len(testloader))
    print("Test Result:")
    print('recall: %.4f, specificity: %.4f, precision: %.4f, F1_score:%.4f, acc: %.4f, iou: %.4f, mae: %.4f, dice: %.4f, hd: %.4f, auc: %.4f'
        % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
           metrics_result['F1_score'],
           metrics_result['acc'], metrics_result['iou'], metrics_result['mae'], metrics_result['dice'],
           metrics_result['hd'], metrics_result['auc']))
    print("total_cost_time:", total_cost_time)
    print("loop_cost_time:", time.time() - all_start)
    evaluation_dir = os.path.sep.join([args.save_dir, args.test_fold + '-' + args.test_dataset + '/'])
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)

    metrics_result['inference_time'] = total_cost_time / len(testloader)
    values_txt = [args.fold, 100 * metrics_result['auc'], 100 * metrics_result['precision'], 100 * metrics_result['acc'], 100 * metrics_result['iou'], 100 * metrics_result['dice'], metrics_result['hd'], 100 * metrics_result['recall'], 100 * metrics_result['specificity'], metrics_result['mae'], 100 * metrics_result['F1_score']]
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
