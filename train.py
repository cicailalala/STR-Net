import argparse
import os
import random
import time

# PyTorch includes
import torch
import torch.optim as optim
import torch.nn.functional as F

# Tensorboard include
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

# Dataloaders includes
from dataloaders import tn3k, tg3k, tatn, tnus, tatu
from dataloaders import custom_transforms as trforms
from dataloaders import utils

# Model includes

from model.strnet import STRNet

# Loss function includes
from model.utils import soft_dice, soft_mse, boundary_loss
import numpy as np

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')

    ## Model settings
    # strnet
    parser.add_argument('-model_name', type=str, default='unet')
    parser.add_argument('-criterion', type=str, default='Dice')
    parser.add_argument('-pretrain', type=str, default='None')  # THYROID

    parser.add_argument('-num_classes', type=int, default=1)
    parser.add_argument('-input_size', type=int, default=224)
    parser.add_argument('-output_stride', type=int, default=16)

    ## Train settings
    parser.add_argument('-dataset', type=str, default='TATN')  # TATN, TATU 
    parser.add_argument('-fold', type=str, default='0')
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-nepochs', type=int, default=80)
    parser.add_argument('-resume_epoch', type=int, default=0)
    parser.add_argument('-data_root', type=str, default='./data/')
    parser.add_argument('-save_path', type=str, default='./models/')

    ## Optimizer settings
    parser.add_argument('-naver_grad', type=str, default=1)
    parser.add_argument('-lr', type=float, default=1e-2)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-update_lr_every', type=int, default=10)
    parser.add_argument('-weight_decay', type=float, default=5e-4)

    ## Visualization settings
    parser.add_argument('-save_every', type=int, default=10)
    parser.add_argument('-log_every', type=int, default=50)
    parser.add_argument('-load_path', type=str, default='')
    parser.add_argument('-use_test', type=int, default=1)
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1234)


def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, 1, img_x, img_y).cuda()
    mask = torch.ones(batch_size, channel, img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*2/3), int(img_y*2/3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[:, :, w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, :, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def get_part_and_rec_ind(volume_shape, img_mask, nb_chnls):
    bs, c, w, h = volume_shape

    # partition
    rand_loc_ind = torch.argsort(torch.rand(bs, nb_cubes, nb_cubes, nb_cubes), dim=0).cuda()
    cube_part_ind = rand_loc_ind.view(bs, 1, nb_cubes, nb_cubes, nb_cubes)
    cube_part_ind = cube_part_ind.repeat_interleave(c, dim=1)
    cube_part_ind = cube_part_ind.repeat_interleave(w // nb_cubes, dim=2)
    cube_part_ind = cube_part_ind.repeat_interleave(h // nb_cubes, dim=3)
    cube_part_ind = cube_part_ind.repeat_interleave(d // nb_cubes, dim=4)

    # recovery
    rec_ind = torch.argsort(rand_loc_ind, dim=0).cuda()
    cube_rec_ind = rec_ind.view(bs, 1, nb_cubes, nb_cubes, nb_cubes)
    cube_rec_ind = cube_rec_ind.repeat_interleave(nb_chnls, dim=1)
    cube_rec_ind = cube_rec_ind.repeat_interleave(w // nb_cubes, dim=2)
    cube_rec_ind = cube_rec_ind.repeat_interleave(h // nb_cubes, dim=3)
    cube_rec_ind = cube_rec_ind.repeat_interleave(d // nb_cubes, dim=4)

    return cube_part_ind, cube_rec_ind

def sigmoid_rampup(current, rampup_length):
    '''Exponential rampup from https://arxiv.org/abs/1610.02242'''
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch, max_epoch):
    return 0.1 * sigmoid_rampup(epoch, max_epoch)

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    save_dir_root = args.save_path
    save_dir = os.path.join(save_dir_root, args.dataset, f"{args.model_name}_fold{args.fold}")
    log_dir = os.path.join(save_dir, 'log')
    writer = SummaryWriter(log_dir=log_dir)
    batch_size = args.batch_size


    net = STRNet(in_ch=3, out_ch=1)
    ema_net = STRNet(in_ch=3, out_ch=1)
    for param in ema_net.parameters():
        param.detach_()
    batch_size = 8
        

    if args.resume_epoch == 0:
        print('Training ' + args.model_name + ' from scratch...')
    else:
        load_path = os.path.join(save_dir, args.model_name + '_epoch-' + str(args.resume_epoch) + '.pth')
        print('Initializing weights from: {}...'.format(load_path))
        net.load_state_dict(torch.load(load_path))

    if args.pretrain == 'THYROID':
        net.load_state_dict(
            torch.load('./run/unet_gland_pretrain/unet_best.pth', map_location=lambda storage, loc: storage))
        print('loading pretrain model......')

    torch.cuda.set_device(device=0)
    net.cuda()
    if args.model_name == 'strnet':
        ema_net.cuda()
        ema_net.train()

    optimizer = optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    if args.criterion == 'Dice':
        criterion = soft_dice
    else:
        raise NotImplementedError

    composed_transforms_tr = transforms.Compose([
        trforms.FixedResize(size=(int(args.input_size), int(args.input_size))),
        trforms.RandomHorizontalFlip(),
        trforms.RandomVerticalFlip(),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])


    if args.dataset == 'TATN':
        train_data = tatn.TATN(mode='train', transform=composed_transforms_tr, fold=args.fold, root=args.data_root)
        val_data = tatn.TATN(mode='val', transform=composed_transforms_ts, fold=args.fold, root=args.data_root)
    elif args.dataset == 'TATU':
        train_data = tatu.TATU(mode='train', transform=composed_transforms_tr, fold=args.fold, root=args.data_root)
        val_data = tatu.TATU(mode='val', transform=composed_transforms_ts, fold=args.fold, root=args.data_root)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
                             pin_memory=True)
    testloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    num_iter_tr = len(trainloader)
    num_iter_ts = len(testloader)
    nitrs = args.resume_epoch * num_iter_tr
    nsamples = args.resume_epoch * len(train_data)
    print('nitrs: %d num_iter_tr: %d' % (nitrs, num_iter_tr))
    print('nsamples: %d tot_num_samples: %d' % (nsamples, len(train_data)))

    log_txt = open(log_dir + '/log.txt', 'w')
    aveGrad = 0
    global_step = 0
    best_flag = 0.0
    recent_losses = []
    nodule_losses = []
    thyroid_losses = []
    start_t = time.time()

    for epoch in range(args.resume_epoch, args.nepochs):
        net.train()
        epoch_losses = []
        epoch_nodule_losses = []
        epoch_thyroid_losses = []
        epoch_mse_losses = []
        consistency_weight = get_current_consistency_weight(epoch, args.nepochs)
        for ii, sample_batched in enumerate(trainloader):
            if args.model_name == 'strnet':
                nodules, glands = sample_batched
                scale = nodules['scale'].cuda()
                inputs_n, labels_n = nodules['image'].cuda(), nodules['label'].cuda()
                inputs_g, labels_g = glands['image'].cuda(), glands['label'].cuda()
                inputs = torch.cat([inputs_n[0].unsqueeze(0), inputs_g[0].unsqueeze(0)], dim=0)

                for i in range(1, inputs_n.size()[0]):
                    inputs = torch.cat([inputs, inputs_n[i].unsqueeze(0)], dim=0)
                    inputs = torch.cat([inputs, inputs_g[i].unsqueeze(0)], dim=0)
                
                # print(inputs_n.shape, labels_n.shape, inputs_g.shape, labels_g.shape, inputs.shape)
                
                
                img_mask, loss_mask = generate_mask(inputs_n)    # boundary 1 center 0
                img_nodules_glands = inputs_n * img_mask + inputs_g * (1 - img_mask) # boundary n center g
                img_glands_nodules = inputs_n * (1 - img_mask) + inputs_g * img_mask # boundary g center n                
                
                # gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)                
                global_step += inputs.data.shape[0]
                inputs = torch.cat([inputs, img_nodules_glands], dim=0)
                inputs = torch.cat([inputs, img_glands_nodules], dim=0)

                nodule, thyroid, pred_scale, thyroid_8, thyroid_4, thyroid_2, nodule_8, nodule_4, nodule_2, nodule_1 = net.forward(inputs)

                # Get pseudo-label from teacher model
                noise = torch.clamp(torch.randn_like(inputs) * 0.1, -0.2, 0.2)
                ema_inputs = inputs + noise
                with torch.no_grad():
                    ema_nodule, ema_thyroid, ema_pred_scale, ema_thyroid_8, ema_thyroid_4, ema_thyroid_2, ema_nodule_8, ema_nodule_4, ema_nodule_2, ema_nodule_1 = ema_net(ema_inputs)
                    ema_nodule_soft = ema_nodule
                    ema_thyroid_soft = ema_thyroid
                    ema_nodule_mix_thyroid_soft = ema_nodule_soft[batch_size*2:batch_size*3] * (1 - loss_mask) + ema_nodule_soft[batch_size*3:] * loss_mask
                    ema_thyroid_mix_nodule_soft = ema_thyroid_soft[batch_size*2:batch_size*3] * loss_mask + ema_thyroid_soft[batch_size*3:] * (1 - loss_mask)
                # print(nodule.shape, thyroid.shape, pred_scale.shape)
                nodule_mix = nodule[batch_size*2:]
                nodule_mix_thyroid = nodule_mix[:batch_size] * (1 - loss_mask) + nodule_mix[batch_size:] * loss_mask
                nodule_mix = nodule_mix[:batch_size] * loss_mask + nodule_mix[batch_size:] * (1 - loss_mask)
                nodule = nodule[:batch_size*2]
                thyroid_mix = thyroid[batch_size*2:]
                thyroid_mix_nodule = thyroid_mix[:batch_size] * loss_mask + thyroid_mix[batch_size:] * (1 - loss_mask)
                thyroid_mix = thyroid_mix[:batch_size] * (1 - loss_mask) + thyroid_mix[batch_size:] * loss_mask
                thyroid = thyroid[:batch_size*2]
                pred_scale = pred_scale[:batch_size*2]
                
                # print(nodule.shape, thyroid.shape, pred_scale.shape, nodule_mix.shape, thyroid_mix.shape)
                pred_scales = torch.zeros(int(len(pred_scale) / 2))
                for i in range(len(pred_scales)):
                    pred_scales[i] = pred_scale[i * 2]
                

                
                
                loss = 0
                nodule_loss_mini = 0
                thyroid_loss_mini = 0
                mse_loss_mini = 0
                consistency_loss = 0
                nodule_soft = nodule
                thyroid_soft = thyroid
                nodule_mix_thyroid_soft = nodule_mix_thyroid
                thyroid_mix_nodule_soft = thyroid_mix_nodule                              
                for i in range(batch_size*2):
                    if i % 2 == 0:
                        nodule_loss = criterion(nodule[i], labels_n[int(i / 2)]) + criterion(nodule_1[i], labels_n[int(i / 2)]) + 0.75*criterion(nodule_2[i], labels_n[int(i / 2)]) + 0.5*criterion(nodule_4[i], labels_n[int(i / 2)]) + 0.25*criterion(nodule_8[i], labels_n[int(i / 2)])
                        nodule_loss_mini += nodule_loss
                        consistency_loss += torch.mean((thyroid_soft[i] - ema_thyroid_soft[i])**2) 
                    else:
                        thyroid_loss = criterion(thyroid[i], labels_g[int((i - 1) / 2)]) + 0.75*criterion(thyroid_2[i], labels_g[int((i - 1) / 2)]) + 0.5*criterion(thyroid_4[i], labels_g[int((i - 1) / 2)]) + 0.25*criterion(thyroid_8[i], labels_g[int((i - 1) / 2)])
                        thyroid_loss_mini += thyroid_loss
                        consistency_loss += torch.mean((nodule_soft[i] - ema_nodule_soft[i])**2)
                for i in range(batch_size):
                    nodule_loss = criterion(nodule_mix[i], labels_n[i])
                    nodule_loss_mini += nodule_loss

                    thyroid_loss = 1 * criterion(thyroid_mix[i], labels_g[i])
                    thyroid_loss_mini += thyroid_loss
                    
                    consistency_loss += torch.mean((nodule_mix_thyroid_soft[i] - ema_nodule_mix_thyroid_soft[i])**2)
                    consistency_loss += torch.mean((thyroid_mix_nodule_soft[i] - ema_thyroid_mix_nodule_soft[i])**2)                        
                        

                mse_loss_mini = (1 - epoch / args.nepochs) * soft_mse(pred_scales.cuda(), scale.float())
                loss += mse_loss_mini
                log_txt.write(str(round(nodule_loss_mini.item(), 3)) + ' ' + str(
                    round(thyroid_loss_mini.item(), 3)) + ' ' + str(round(mse_loss_mini.item(), 3)) + '\n')

                loss += (nodule_loss_mini + thyroid_loss_mini)
                loss += consistency_weight * consistency_loss / 8

            trainloss = loss.item()
            epoch_losses.append(trainloss)
            if len(recent_losses) < args.log_every:
                recent_losses.append(trainloss)
            else:
                recent_losses[nitrs % len(recent_losses)] = trainloss

            # Backward the averaged gradient
            loss.backward()
            aveGrad += 1
            nitrs += 1
            nsamples += args.batch_size

            # Update the weights once in p['nAveGrad'] forward passes
            if aveGrad % args.naver_grad == 0:
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

            if nitrs % args.log_every == 0:
                meanloss = sum(recent_losses) / len(recent_losses)
                print('epoch: %d ii: %d trainloss: %.2f timecost:%.2f secs' % (
                    epoch, ii, meanloss, time.time() - start_t))
                writer.add_scalar('data/trainloss', meanloss, nsamples)
                if 'trfe' in args.model_name or args.model_name == 'mtnet':
                    writer.add_scalar('data/train_nodule_loss', sum(nodule_losses) / len(nodule_losses), nsamples)
                    writer.add_scalar('data/train_thyroid_loss', sum(thyroid_losses) / len(thyroid_losses), nsamples)

        meanloss = sum(epoch_losses) / len(epoch_losses)
        print('epoch: %d meanloss: %.2f' % (epoch, meanloss))
        writer.add_scalar('data/epochloss', meanloss, nsamples)

        if args.use_test == 1:
            prec_lists = []
            recall_lists = []
            sum_testloss = 0.0
            total_mae = 0.0
            count = 0
            iou = 0
            net.eval()
            for ii, sample_batched in enumerate(testloader):
                inputs, labels = sample_batched['image'].cuda(), sample_batched['label'].cuda()
                with torch.no_grad():
                    outputs, _, _, _, _, _, _, _, _, _  = net.forward(inputs)

                sum_testloss += loss.item()


                predictions = outputs

                iou += utils.get_iou(predictions, labels)
                count += 1

                total_mae += utils.get_mae(predictions, labels) * predictions.size(0)
                prec_list, recall_list = utils.get_prec_recall(predictions, labels)
                prec_lists.extend(prec_list)
                recall_lists.extend(recall_list)

                if ii % num_iter_ts == num_iter_ts - 1:
                    mmae = total_mae / count
                    mean_testloss = sum_testloss / num_iter_ts
                    mean_prec = sum(prec_lists) / len(prec_lists)
                    mean_recall = sum(recall_lists) / len(recall_lists)
                    fbeta = 1.3 * mean_prec * mean_recall / (0.3 * mean_prec + mean_recall)
                    iou = iou / count

                    print('Validation:')
                    print('epoch: %d, numImages: %d testloss: %.2f mmae: %.4f fbeta: %.4f iou: %.4f' % (
                        epoch, count, mean_testloss, mmae, fbeta, iou))
                    writer.add_scalar(f'data/{args.dataset}_validloss', mean_testloss, nsamples)
                    writer.add_scalar(f'data/{args.dataset}_validmae', mmae, nsamples)
                    writer.add_scalar(f'data/{args.dataset}_validfbeta', fbeta, nsamples)
                    writer.add_scalar(f'data/{args.dataset}_validiou', iou, epoch)

                    if iou > best_flag:
                        save_path = os.path.join(save_dir, args.model_name + '_best' + '.pth')
                        if not os.path.exists(os.path.dirname(save_path)):
                            os.makedirs(os.path.dirname(save_path))
                        torch.save(net.state_dict(), save_path)
                        print("Save model at {}\n".format(save_path))
                        best_flag = iou


        if epoch % args.update_lr_every == args.update_lr_every - 1:
            lr_ = utils.lr_poly(args.lr, epoch, args.nepochs, 0.9)
            print('(poly lr policy) learning rate: ', lr_)
            optimizer = optim.SGD(
                net.parameters(),
                lr=lr_,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )
    writer.close()


if __name__ == "__main__":
    args = get_arguments()
    main(args)



      
        
        