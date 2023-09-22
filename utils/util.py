import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from skimage.morphology import skeletonize
from PIL import Image


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def setSeed(config):
    if config["seed"] is None:
        manualSeed = np.random.randint(1, 10000)
    else:
        manualSeed = config["seed"]
    print("Random Seed: ", manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)


def getParllelNetworkStateDict(state_dict):
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def to_variable(tensor, volatile=False, requires_grad=False):
    return Variable(tensor.long().cuda(), requires_grad=requires_grad)


def weights_init(model, manual_seed=7):
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    random.seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def weights_normal_init(model, manual_seed=7):
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    random.seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


def performAngleMetrics(
    train_loss_angle_file, val_loss_angle_file, epoch, hist, is_train=True, write=False
):

    pixel_accuracy = np.diag(hist).sum() / hist.sum()
    mean_accuracy = np.diag(hist) / hist.sum(1)
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    freq = hist.sum(1) / hist.sum()
    fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
    if write and is_train:
        train_loss_angle_file.write(
            "[%d], Pixel Accuracy:%.3f, Mean Accuracy:%.3f, Mean IoU:%.3f, Freq.Weighted Accuray:%.3f  \n"
            % (
                epoch,
                100 * pixel_accuracy,
                100 * np.nanmean(mean_accuracy),
                100 * np.nanmean(iou),
                100 * fwavacc,
            )
        )
    elif write and not is_train:
        val_loss_angle_file.write(
            "[%d], Pixel Accuracy:%.3f, Mean Accuracy:%.3f, Mean IoU:%.3f, Freq.Weighted Accuray:%.3f  \n"
            % (
                epoch,
                100 * pixel_accuracy,
                100 * np.nanmean(mean_accuracy),
                100 * np.nanmean(iou),
                100 * fwavacc,
            )
        )

    return 100 * pixel_accuracy, 100 * np.nanmean(iou), 100 * fwavacc

def performAngleMetrics_Test(
   test_loss_angle_file, hist,write=True
):

    pixel_accuracy = np.diag(hist).sum() / hist.sum()
    mean_accuracy = np.diag(hist) / hist.sum(1)
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    freq = hist.sum(1) / hist.sum()
    fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
    if write:
        test_loss_angle_file.write(
            "Pixel Accuracy:%.3f, Mean Accuracy:%.3f, Mean IoU:%.3f, Freq.Weighted Accuray:%.3f  \n"
             % (
                100 * pixel_accuracy,
                100 * np.nanmean(mean_accuracy),
                100 * np.nanmean(iou),
                100 * fwavacc,
             )
         )

    return 100 * np.nanmean(pixel_accuracy), 100 * np.nanmean(iou), 100 * fwavacc


def performMetrics(
    train_loss_file,
    val_loss_file,
    epoch,
    hist,
    loss,
    loss_vec,
    is_train=True,
    write=False,
):

    pixel_accuracy = np.diag(hist).sum() / hist.sum()
    mean_accuracy = np.diag(hist) / hist.sum(1)
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    class_iou = iou[1:]
    # class_iou = iou[1]
    freq = hist.sum(1) / hist.sum()
    fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()

    if write and is_train:
        train_loss_file.write(
            "[%d], Loss:%.5f, Loss(VecMap):%.5f, Pixel Accuracy:%.3f, Mean Accuracy:%.3f, Mean IoU:%.3f, Class IoU:%.5f, Railway IoU:%.3f, Driveway IoU:%.3f, Pathway IoU:%.3f, Bridge IoU:%.3f, Freq.Weighted Accuray:%.3f  \n"
            % (
                epoch,
                loss,
                loss_vec,
                100 * pixel_accuracy,
                100 * np.nanmean(mean_accuracy),
                100 * np.nanmean(iou),
                100*np.nanmean(class_iou),
                100 * iou[1],
                100 * iou[2],
                100 * iou[3],
                100 * iou[4],
                100 * fwavacc,
            )
        )
    elif write and not is_train:
        val_loss_file.write(
            "[%d], Loss:%.5f, Loss(VecMap):%.5f, Pixel Accuracy:%.3f, Mean Accuracy:%.3f, Mean IoU:%.3f, Class IoU:%.5f, Railway IoU:%.3f, Driveway IoU:%.3f, Pathway IoU:%.3f, Bridge IoU:%.3f, Freq.Weighted Accuray:%.3f  \n"
            % (
                epoch,
                loss,
                loss_vec,
                100 * pixel_accuracy,
                100 * np.nanmean(mean_accuracy),
                100 * np.nanmean(iou),
                100*np.nanmean(class_iou),
                100 * iou[1],
                100 * iou[2],
                100 * iou[3],
                100 * iou[4],
                100 * fwavacc,
            )
        )

    return (
        100 * np.nanmean(pixel_accuracy),
        100 * np.nanmean(iou),
        100 * np.nanmean(class_iou),
        100 * iou[1],
        100 * iou[2],
        100 * iou[3],
        100 * iou[4],
        100 * fwavacc,
    )

def performMetrics_2(
    train_loss_file,
    val_loss_file,
    epoch,
    hist,
    loss,
    loss_vec,
    is_train=True,
    write=False,
):

    pixel_accuracy = np.diag(hist).sum() / hist.sum()
    mean_accuracy = np.diag(hist) / hist.sum(1)
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    class_iou = iou[1:]
    # class_iou = iou[1]
    freq = hist.sum(1) / hist.sum()
    fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()

    if write and is_train:
        train_loss_file.write(
            "[%d], Loss:%.5f, Loss(VecMap):%.5f, Pixel Accuracy:%.3f, Mean Accuracy:%.3f, Mean IoU:%.3f, Class IoU:%.5f, Freq.Weighted Accuray:%.3f  \n"
            % (
                epoch,
                loss,
                loss_vec,
                100 * pixel_accuracy,
                100 * np.nanmean(mean_accuracy),
                100 * np.nanmean(iou),
                100*np.nanmean(class_iou),
                100 * fwavacc,
            )
        )
    elif write and not is_train:
        val_loss_file.write(
            "[%d], Loss:%.5f, Loss(VecMap):%.5f, Pixel Accuracy:%.3f, Mean Accuracy:%.3f, Mean IoU:%.3f, Class IoU:%.5f, Freq.Weighted Accuray:%.3f  \n"
            % (
                epoch,
                loss,
                loss_vec,
                100 * pixel_accuracy,
                100 * np.nanmean(mean_accuracy),
                100 * np.nanmean(iou),
                100*np.nanmean(class_iou),
                100 * fwavacc,
            )
        )

    return (
        100 * np.nanmean(pixel_accuracy),
        100 * np.nanmean(iou),
        100 * np.nanmean(class_iou),
        100 * fwavacc,
    )

def performMetrics_Test(
    test_loss_file,
    hist,
    write=True,
):

    pixel_accuracy = np.diag(hist).sum() / hist.sum()
    mean_accuracy = np.diag(hist) / hist.sum(1)
    mean_accuracy = mean_accuracy[1:]
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    class_iou = iou[1:]
    # freq = hist.sum(1) / hist.sum()
    freq = hist[1:].sum(1) / hist[1:].sum()
    # fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
    fwavacc = (freq[freq > 0] * class_iou[freq > 0]).sum()

    if write:
        test_loss_file.write(
            "Pixel Accuracy:%.3f, Mean Accuracy:%.3f, Mean IoU:%.3f, Class IoU:%.5f, Railway IoU:%.3f, Driveway IoU:%.3f, Pathway IoU:%.3f, Bridge IoU:%.3f, Freq.Weighted Accuray:%.3f  \n"
             % (
                100 * pixel_accuracy,
                100 * np.nanmean(mean_accuracy),
                100 * np.nanmean(iou),
                100*np.nanmean(class_iou),
                100 * iou[1],
                100 * iou[2],
                100 * iou[3],
                100 * iou[4],
                100 * fwavacc
            )
        )

    return (
        100 * np.nanmean(pixel_accuracy),
        100 * np.nanmean(iou),
        100 * np.nanmean(class_iou),
        100 * iou[1],
        100 * iou[2],
        100 * iou[3],
        100 * iou[4],
        100 * fwavacc,
    )

def performMetrics_Test_2(
    test_loss_file,
    hist,
    write = True
):

    pixel_accuracy = np.diag(hist).sum() / hist.sum()
    mean_accuracy = np.diag(hist) / hist.sum(1)
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    class_iou = iou[1:]
    # freq = hist.sum(1) / hist.sum()
    freq = hist[1:].sum(1) / hist[1:].sum()
    # fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
    fwavacc = (freq[freq > 0] * class_iou[freq > 0]).sum()

    if write:
        test_loss_file.write(
            "Pixel Accuracy:%.3f, Mean Accuracy:%.3f, Mean IoU:%.3f, Class IoU:%.5f, Freq.Weighted Accuray:%.3f  \n"
             % (
                100 * pixel_accuracy,
                100 * np.nanmean(mean_accuracy),
                100 * np.nanmean(iou),
                100*np.nanmean(class_iou),
                100 * fwavacc
            )
        )

    return (
        100 * pixel_accuracy,
        100 * np.nanmean(iou),
        100 * np.nanmean(class_iou),
        100 * fwavacc,
    )

def fast_hist(a, b, n):

    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def save_checkpoint(epoch, loss, model, optimizer, best_accuracy, best_miou, config, experiment_dir):

    if torch.cuda.device_count() > 1:
        arch = type(model.module).__name__
    else:
        arch = type(model).__name__
    state = {
        "arch": arch,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "pixel_accuracy": best_accuracy,
        "miou": best_miou,
        "config": config,
    }
    filename = os.path.join(
        experiment_dir, "checkpoint-epoch{:03d}-loss-{:.4f}.pth.tar".format(
            epoch, loss)
    )
    torch.save(state, filename)
    if os.path.exists(os.path.join(experiment_dir, "model_best.pth.tar")):
        os.remove(os.path.join(experiment_dir, "model_best.pth.tar"))
        os.rename(filename, os.path.join(experiment_dir, "model_best.pth.tar"))
    else:
        os.rename(filename, os.path.join(experiment_dir, "model_best.pth.tar"))
    print("Saving current best: {} ...".format("model_best.pth.tar"))




def savePredictedProb_MTL_5(
    real,
    gt,
    predicted,
    pred_affinity=None,
    image_name="",
    road_name="",
    angle_name="",
    norm_type="Mean",
    crop_size=256
):
    b, c, h, w = real.size()
    grid = []
    mean_bgr = np.array([0.34509784, 0.33666852, 0.34015575])
    deviation_bgr = np.array([0.17548144, 0.16095427, 0.16002631])

    for idx in range(b):
        # real_ = np.asarray(real[idx].numpy().transpose(1,2,0),dtype=np.float32)
        real_ = np.asarray(real[idx].numpy().transpose(
            1, 2, 0), dtype=np.float32)
        if norm_type == "Mean":
            real_ = real_ + mean_bgr
        elif norm_type == "Std":
            real_ = (real_ * deviation_bgr) + mean_bgr

        real_ = np.asarray(real_, dtype=np.uint8)

        # 设置调色板
        color = {0: [255, 255, 255],  # 0=白色（背景）
                 1: [255, 0, 0],  # 1=红色（铁路）
                 2: [230, 152, 0],  # 2=土黄（公路）
                 3: [0, 0, 0],  # 3=黑色（小路）
                 4: [0, 0, 255]  # 4=蓝色（桥梁）
                 }

        #保存组合图中的道路真相
        predict_gt = np.zeros((crop_size,crop_size,3)) # 转RGB用的，接收相应的预测RGB值
        gt_1 = gt[idx].numpy()

        # predict_gt是用来赋色的：color[i] 代表一类的颜色值；colors[i][0] 代表该类的 0 通道的值
        for i in range(len(color)):
            predict_gt[:, :, 0] += ((gt_1[:, :] == i) * (color[i][2])).astype('uint8')
            predict_gt[:, :, 1] += ((gt_1[:, :] == i) * (color[i][1])).astype('uint8')
            predict_gt[:, :, 2] += ((gt_1[:, :] == i) * (color[i][0])).astype('uint8')

        gt_ = predict_gt

        # predicted_ = (predicted[idx]).numpy() * 255.0

        #保存组合图中的道路预测结果

        color = {0:[255, 255, 255],  # 0=白色（背景）
                 1:[255, 0, 0],  # 1=红色（铁路）
                 2:[230, 152, 0],  # 2=土黄（公路）
                 3:[0, 0, 0],  # 3=黑色（小路）
                 4:[0, 0, 255]  # 4=蓝色（桥梁）
                 }
        predict_road = np.zeros((crop_size,crop_size,3)) # 转RGB用的，接收相应的预测RGB值
        predicted_1 = (predicted[idx]).numpy()   #网络输出的二维数组值

        # predict_road 是用来赋色的：color[i] 代表一类的颜色值；colors[i][0] 代表该类的 0 通道的值
        for i in range(len(color)):
            predict_road[:, :, 0] += ((predicted_1[:, :] == i) * (color[i][2])).astype('uint8')
            predict_road[:, :, 1] += ((predicted_1[:, :] == i) * (color[i][1])).astype('uint8')
            predict_road[:, :, 2] += ((predicted_1[:, :] == i) * (color[i][0])).astype('uint8')


        predicted_ = predict_road



        #单独保存道路预测结果
        predicted_single = (predicted[idx]).numpy()
        predicted_single = np.asarray(predicted_single, dtype=np.uint8)
        road_name1 = os.path.join(road_name+"_{}.png".format(idx))
        cv2.imwrite(road_name1, predicted_single)


        # predicted_prob_ = (predicted_prob[idx]).numpy() * 255.0
        #
        # # predicted_prob_ = (predicted_prob[idx]).numpy()
        # # print(predicted_prob_)
        #
        # # predicted_prob_ = predicted_prob_[:,:]
        # predicted_prob_ = np.asarray(predicted_prob_, dtype=np.uint8)
        # # predicted_prob_ = np.stack((predicted_prob_,)*3).transpose(1,2,0)
        # predicted_prob_ = cv2.applyColorMap(predicted_prob_, cv2.COLORMAP_JET)

        # array = np.random.randint(0, 35, size=(512, 512))
        # affinity_1 = np.asarray(array, dtype=np.uint8)
        #
        # im = Image.fromarray(affinity_1)
        # im.convert('L').save(angle_name)  # 保存为灰度图(8-bit)



        if pred_affinity is not None:
            hsv = np.zeros_like(real_)
            hsv[..., 1] = 255
            affinity_ = pred_affinity[idx].numpy()


            mag = np.copy(affinity_)
            mag[mag < 36] = 1
            mag[mag >= 36] = 0
            affinity_[affinity_ == 36] = 0


            hsv[..., 0] = affinity_ * 10 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            affinity_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # 单独保存角度预测结果
            affinity_single = (pred_affinity[idx]).numpy()
            affinity_single = np.asarray(affinity_single, dtype=np.uint8)
            angle_name1 = os.path.join(angle_name + "_{}.png".format(idx))
            cv2.imwrite(angle_name1, affinity_single)



            pair = np.concatenate(
                (real_, gt_, predicted_, affinity_bgr), axis=1
            )
        else:
            pair = np.concatenate(
                (real_, gt_, predicted_), axis=1)
        grid.append(pair)

    if pred_affinity is not None:
        cv2.imwrite(image_name, np.array(grid).reshape(b * h, 4 * w, 3))
    else:
        cv2.imwrite(image_name, np.array(grid).reshape(b * h, 3 * w, 3))


def get_relaxed_precision(a, b, buffer):
    tp = 0
    indices = np.where(a == 1)
    for ind in range(len(indices[0])):
        tp += (np.sum(
            b[indices[0][ind]-buffer: indices[0][ind]+buffer+1,
              indices[1][ind]-buffer: indices[1][ind]+buffer+1]) > 0).astype(np.int)
    return tp


def relaxed_f1(pred, gt, buffer=3):
    ''' Usage and Call
    # rp_tp, rr_tp, pred_p, gt_p = relaxed_f1(predicted.cpu().numpy(), labels.cpu().numpy(), buffer = 3)

    # rprecision_tp += rp_tp
    # rrecall_tp += rr_tp
    # pred_positive += pred_p
    # gt_positive += gt_p

    # precision = rprecision_tp/(gt_positive + 1e-12)
    # recall = rrecall_tp/(gt_positive + 1e-12)
    # f1measure = 2*precision*recall/(precision + recall + 1e-12)
    # iou = precision*recall/(precision+recall-(precision*recall) + 1e-12)
    '''

    rprecision_tp, rrecall_tp, pred_positive, gt_positive = 0, 0, 0, 0
    for b in range(pred.shape[0]):
        pred_sk = skeletonize(pred[b])
        gt_sk = skeletonize(gt[b])
        # pred_sk = pred[b]
        # gt_sk = gt[b]
        rprecision_tp += get_relaxed_precision(pred_sk, gt_sk, buffer)
        rrecall_tp += get_relaxed_precision(gt_sk, pred_sk, buffer)
        pred_positive += len(np.where(pred_sk == 1)[0])
        gt_positive += len(np.where(gt_sk == 1)[0])

    return rprecision_tp, rrecall_tp, pred_positive, gt_positive
