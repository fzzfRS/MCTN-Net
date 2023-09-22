from __future__ import print_function

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from model.models import MODELS
from dataset_preprocess import MCTNDataset
from torch.autograd import Variable
from utils import util
from utils import viz_util
os.environ['CUDA_VISIBLE_DEVICES']='0'

__dataset__ = {"MCTNDataset": MCTNDataset}



parser = argparse.ArgumentParser()

#设置参数文件路径
parser.add_argument(
    "--config", default='./config.json', type=str, help="config file path"
)

#设置模型名称
parser.add_argument(
    "--model_name",
    default='MCTNNet',
    help="Name of Model =")

#设置 实验名称
parser.add_argument("--exp", default='road', type=str, help="Experiment Name/Directory")

#设置预测结果保存路径
parser.add_argument("--result", default='road', type=str, help="path to predict result")

#设置网络权重文件地址
parser.add_argument("--resume", default=r'./checkpoint/MCTN/road/model_best.pth.tar' ,type=str, help="path to best checkpoint"
)

#设置数据集名称
parser.add_argument(
    "--dataset",
    default= 'spacenet',type=str,
    help="select dataset name "
)

#设置模型参数
parser.add_argument(
    "--model_kwargs",
    default={},
    type=json.loads,
    help="parameters for the model",
)

#设置是否执行多尺度预测
parser.add_argument(
    "--multi_scale_pred",
    default=True,
    type=util.str2bool,
    help="perform multi-scale prediction (default: True)",
)

#将设置好的参数赋值给args
args = parser.parse_args()

# 导入设定的测试集路径等参数

if args.config is not None:
    config = json.load(open(args.config))

assert config is not None #如果参数文件不是空，则继续往下运行

util.setSeed(config) #设置随机种子数

print(args.resume)
#结果保存路径
result_dir = os.path.join(config["tester"]["save_dir"], args.result)

#确认路径，如果没有则创建
util.ensure_dir(result_dir)


###Logging Files（实验结果保存日志）

test_file = "{}/{}_test_road.txt".format(result_dir, args.dataset)

#打开实验结果保存文件，设置文件写入方式为只写

test_result = open(test_file, "w")

### Angle Metrics（角度实验结果保存日志）
test_file_angle = "{}/{}_test_angle.txt".format(result_dir, args.dataset)

#打开角度实验结果保存文件，设置文件写入方式为只写
test_angle_result = open(test_file_angle, "w")
################################################################################


#搭建模型，包括模型选择以及两个任务分类数的参数传入
model = MODELS[args.model_name](
    config["task1_classes"], **args.model_kwargs
)
model = nn.DataParallel(model).cuda()


### Load Dataset from root folder and intialize DataLoader

test_loader = data.DataLoader(
    __dataset__[args.dataset](
        config["test_dataset"],
        seed=config["seed"],
        is_train=False,
        is_test=True,
        multi_scale_pred=args.multi_scale_pred,
    ),

    batch_size=config["test_batch_size"],
    num_workers=0,
    shuffle=False,
    pin_memory=False,
)
print("Testing with dataset => {}".format(test_loader.dataset.__class__.__name__))
################################################################################


if args.resume is not None:
    print("Loading from {} and predicting....".format(args.model_name))
    checkpoint = torch.load(args.resume,map_location='cpu')#加载保存的模型文件
    model.load_state_dict(checkpoint["state_dict"]) #加载字典


#统计模型参数
viz_util.summary(model, print_arch=False)

def test():
    model.eval()
    hist = np.zeros((config["task1_classes"], config["task1_classes"]))
    hist_angles = np.zeros((config["task2_classes"], config["task2_classes"]))
    crop_size = config["test_dataset"][args.dataset]["crop_size"]
    for i, (inputsBGR, labels, vecmap_angles) in enumerate(test_loader, 0):

        with torch.no_grad():
            inputsBGR = Variable(
                inputsBGR.float().cuda(), volatile=True, requires_grad=False
            )

            outputs, pred_vecmaps = model(inputsBGR)
            outputs = outputs[-1]
            pred_vecmaps = pred_vecmaps[-1]

            _, predicted = torch.max(outputs.data, 1)

            correctLabel = labels[-1].view(-1, crop_size, crop_size).long()
            hist += util.fast_hist(
                predicted.view(predicted.size(0), -1).cpu().numpy(),
                correctLabel.view(correctLabel.size(0), -1).cpu().numpy(),
                config["task1_classes"],
            )

            _, predicted_angle = torch.max(pred_vecmaps.data, 1)
            correct_angles = vecmap_angles[-1].view(-1, crop_size, crop_size).long()
            hist_angles += util.fast_hist(
                predicted_angle.view(predicted_angle.size(0), -1).cpu().numpy(),
                correct_angles.view(correct_angles.size(0), -1).cpu().numpy(),
                config["task2_classes"],
            )

        p_accu, miou, road_iou, railway_iou, driveway_iou, pathway_iou, bridge_iou, fwacc = util.performMetrics_Test(
                test_result,
                hist,
                write = False,
            )

        p_accu_angle, miou_angle, fwacc_angle = util.performAngleMetrics_Test(
                test_angle_result,
                hist_angles,
                write = False,
            )


        viz_util.progress_bar(
            i,
            len(test_loader),
            "mIoU: %.4f%%(%.4f%%) | railway:%.4f%% | roadway:%.4f%% | trail:%.4f%% | bridge:%.4f%% | angle miou: %.4f%%"
            % (
                miou,
                road_iou,
                railway_iou,
                driveway_iou,
                pathway_iou,
                bridge_iou,
                miou_angle,
            ),
        )


        images_path = "{}/images/".format(result_dir)
        util.ensure_dir(images_path)
        road_path = "{}/road/".format(result_dir)
        util.ensure_dir(road_path)
        angle_path = "{}/angle/".format(result_dir)
        util.ensure_dir(angle_path)
        util.savePredictedProb_MTL_5(
            inputsBGR.data.cpu(),  # 输入图像
            labels[-1].cpu(),  # 输入标签
            predicted.cpu(),  # 预测图像
            predicted_angle.cpu(),
            os.path.join(images_path, "test_pair_{}.png".format(i)),
            os.path.join(road_path, "road_predict_{}".format( i)),
            os.path.join(angle_path, "angle_predict_{}".format(i)),
            norm_type=config["test_dataset"]["normalize_type"],
            crop_size=crop_size
        )

    del inputsBGR, labels, predicted, outputs,correctLabel,correct_angles

    util.performMetrics_Test(
        test_result,
        hist,
        write=True,
    )

    util.performAngleMetrics_Test(
        test_angle_result,
        hist_angles,
        write=True,
    )


test()
