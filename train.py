from __future__ import print_function

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from model.models import MODELS
from dataset_preprocess import MCTNDataset
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from utils.loss import CrossEntropyLoss2d, WeightedmIoULoss
from utils import util
from utils import viz_util
os.environ['CUDA_VISIBLE_DEVICES']='0'

__dataset__ = {"MCTNDataset": MCTNDataset}


parser = argparse.ArgumentParser()

#设置参数文件路径
parser.add_argument(
    "--config", default=r'./config.json', type=str, help="config file path"
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

#设置继续训练的网络权重文件地址
parser.add_argument("--resume", default=None ,type=str, help="path to latest checkpoint (default: None)"
)

#设置数据集名称
parser.add_argument(
    "--dataset",
    default= 'MCTNDataset',type=str,
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

#初始化参数文件
config = None

if args.resume is not None:
    if args.config is not None:
        print("Warning: --config overridden by --resume")
        config = torch.load(args.resume)["config"]
elif args.config is not None:
    config = json.load(open(args.config))

assert config is not None #如果参数文件不是空，则继续往下运行

util.setSeed(config) #设置随机种子数

#实验路径
experiment_dir = os.path.join(config["trainer"]["save_dir"], args.exp)

#结果保存路径
result_dir = os.path.join(config["tester"]["save_dir"], args.result)

#确认路径，如果没有则创建
util.ensure_dir(experiment_dir)
util.ensure_dir(result_dir)


###Logging Files（实验结果保存日志）
train_file = "{}/{}_train_loss.txt".format(experiment_dir, args.dataset)
val_file = "{}/{}_val_loss.txt".format(experiment_dir, args.dataset)

#打开实验结果保存文件，设置文件写入方式为只写
train_loss_file = open(train_file, "w")
val_loss_file = open(val_file, "w")





### Angle Metrics（角度实验结果保存日志）
train_file_angle = "{}/{}_train_angle_loss.txt".format(experiment_dir, args.dataset)
val_file_angle = "{}/{}_val_angle_loss.txt".format(experiment_dir, args.dataset)


#打开角度实验结果保存文件，设置文件写入方式为只写
train_loss_angle_file = open(train_file_angle, "w")
val_loss_angle_file = open(val_file_angle, "w")
################################################################################

#统计电脑GPU数
num_gpus=1

#搭建模型，包括模型选择以及两个任务分类数的参数传入
model = MODELS[args.model_name](
    config["task1_classes"], config["task2_classes"], **args.model_kwargs
)

#设置使用GPU数
if num_gpus > 1:
    print("Training with multiple GPUs ({})".format(num_gpus))
    model = nn.DataParallel(model).cuda()
else:
    print("Single Cuda Node is avaiable")
    model.cuda()
################################################################################

### Load Dataset from root folder and intialize DataLoader（从根文件夹加载数据集并初始化数据加载器）
train_loader = data.DataLoader(
    __dataset__[args.dataset](
        config["train_dataset"],
        seed=config["seed"],
        is_train=True,
        is_test=False,
        multi_scale_pred=args.multi_scale_pred,
    ),
    batch_size=config["train_batch_size"],
    num_workers=0,
    shuffle=True,
    pin_memory=False,
)

val_loader = data.DataLoader(
    __dataset__[args.dataset](
        config["val_dataset"],
        seed=config["seed"],
        is_train=False,
        is_test=False,
        multi_scale_pred=args.multi_scale_pred,
    ),
    batch_size=config["val_batch_size"],
    num_workers=0,
    shuffle=True,
    pin_memory=False,
)


print("Training with dataset => {}".format(train_loader.dataset.__class__.__name__))
################################################################################

#设置最佳正确率、最大miou，开始轮数，总共训练数据轮数
best_accuracy = 0
best_miou = 0
start_epoch = 1
total_epochs = config["trainer"]["total_epochs"]

#设置优化器
optimizer = optim.SGD(
    model.parameters(), lr=config["optimizer"]["lr"], momentum=0.9, weight_decay=0.0005
)
# optimizer = optim.Adam(model.parameters(), lr=config["optimizer"]["lr"],weight_decay=0.0005)

if args.resume is not None:
    print("Loading from existing FCN and copying weights to continue....")
    checkpoint = torch.load(args.resume)#加载保存的模型文件
    start_epoch = checkpoint["epoch"] + 1   #中断后的epoch开始为中断epoch+1
    best_miou = checkpoint["miou"]          #miou为模型保存的miou
    # stat_parallel_dict = util.getParllelNetworkStateDict(checkpoint['state_dict'])
    # model.load_state_dict(stat_parallel_dict)
    model.load_state_dict(checkpoint["state_dict"]) #加载字典
    optimizer.load_state_dict(checkpoint["optimizer"])  #加载优化器参数
else:
    util.weights_init(model, manual_seed=config["seed"])#权重初始化


#统计模型参数
viz_util.summary(model, print_arch=False)

#学习率衰减器
scheduler = MultiStepLR(
    optimizer,
    milestones=eval(config["optimizer"]["lr_drop_epoch"]),
    gamma=config["optimizer"]["lr_step"],
)
# scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False,
#     threshold=0.01, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)


#设置道路权重，返回一个全为1的"task1_classes"大小的张量矩阵
weights = torch.ones(config["task1_classes"]).cuda()

#判断道路权重，设置权重矩阵
if config["task1_weight"] < 1:
    print("Roads are weighted.")
    weights[0] = 1 - config["task1_weight"]
    weights[1] = config["task1_weight"]

#判断道路角度权重，设置权重矩阵
weights_angles = torch.ones(config["task2_classes"]).cuda()
if config["task2_weight"] < 1:
    print("Road angles are weighted.")
    weights_angles[-1] = config["task2_weight"]


# #设置损失函数
angle_loss = CrossEntropyLoss2d(
    weight=weights_angles, size_average=True, ignore_index=255, reduce=True
).cuda() #ignore_index指定一个被忽略且不影响输入梯度的目标值

road_loss = WeightedmIoULoss(
    weight=weights, size_average=True, n_classes=config["task1_classes"]
).cuda()


def train(epoch):
    #初始化参数
    train_loss_iou = 0  #初始化训练损失为0
    train_loss_vec = 0
    model.train()   #模型设置为train模式
    optimizer.zero_grad()   #优化器梯度清零
    hist = np.zeros((config["task1_classes"], config["task1_classes"]))     #创建道路0矩阵
    hist_angles = np.zeros((config["task2_classes"], config["task2_classes"]))  #创建角度0矩阵
    crop_size = config["train_dataset"][args.dataset]["crop_size"]  #导入裁剪大小


    for i, data in enumerate(train_loader, 0):  #抓取数据与次数，第一次为0和第一个train_loader中数据
        inputsBGR, labels, vecmap_angles =  data   #抓取数据和两个标签
        inputsBGR = Variable(inputsBGR.float().cuda()) #Variable相当于容器，用来装载数据迁移到显存中
        outputs, pred_vecmaps = model(inputsBGR)    #经过模型后产生两个结果

        #多尺度特征计算损失函数
        if args.multi_scale_pred:
            loss1 = road_loss(outputs[0], util.to_variable(labels[0]), False)
            if num_gpus > 1:
                num_stacks = model.module.num_stacks
            else:
                if model.num_stacks  is None:
                    num_stacks = 2
                else:
                    num_stacks =model.num_stacks


            for idx in range(num_stacks - 1):
                loss1 += road_loss(outputs[idx + 1], util.to_variable(labels[0]), False)
            # for idx, output in enumerate(outputs[-2:]):
            #     loss1 += road_loss(output, util.to_variable(labels[idx + 1]), False)
            for idx, output in enumerate(outputs[-3:]):
                if idx <2:
                    loss1 += road_loss(output, util.to_variable(labels[idx + 1]), False)
                else:
                    loss1 += road_loss(output, util.to_variable(labels[-1]), False)
            loss2 = angle_loss(pred_vecmaps[0], util.to_variable(vecmap_angles[0]))
            # loss2 = 0
            for idx in range(num_stacks - 1):
                loss2 += angle_loss(
                    pred_vecmaps[idx + 1], util.to_variable(vecmap_angles[0])
                )
            for idx, pred_vecmap in enumerate(pred_vecmaps[-2:]):
                loss2 += angle_loss(pred_vecmap, util.to_variable(vecmap_angles[idx + 1]))

            outputs = outputs[-1]
            pred_vecmaps = pred_vecmaps[-1]
        else:
            outputs = outputs[-1]
            pred_vecmaps = pred_vecmaps[-1]
            loss1 = road_loss(outputs, util.to_variable(labels[-1]), False)
            loss2 = angle_loss(pred_vecmaps, util.to_variable(vecmap_angles[-1]))

        train_loss_iou += loss1.item()
        train_loss_vec += loss2.item()

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

        p_accu, miou, road_iou, railway_iou, driveway_iou, pathway_iou, bridge_iou, fwacc = util.performMetrics(
            train_loss_file,
            val_loss_file,
            epoch,
            hist,
            train_loss_iou / (i + 1),
            train_loss_vec / (i + 1),
        )


        p_accu_angle, miou_angle, fwacc_angle = util.performAngleMetrics(
            train_loss_angle_file, val_loss_angle_file, epoch, hist_angles, is_train=True
        )

        viz_util.progress_bar(
            i,
            len(train_loader),
            "Loss: %.6f | VecLoss: %.6f | mIoU: %.4f%%(%.4f%%) | rail:%.4f%% | drive:%.4f%% | path:%.4f%% | bridge:%.4f%% | angle miou: %.4f%%"
            % (
                train_loss_iou / (i + 1),
                train_loss_vec / (i + 1),
                miou,
                road_iou,
                railway_iou,
                driveway_iou,
                pathway_iou,
                bridge_iou,
                miou_angle,
            ),
        )

        torch.autograd.backward([loss1, loss2])

        if i % config["trainer"]["iter_size"] == 0 or i == len(train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()

        del (
            outputs,
            pred_vecmaps,
            predicted,
            correct_angles,
            correctLabel,
            inputsBGR,
            labels,
            vecmap_angles,
        )

    util.performMetrics(
        train_loss_file,
        val_loss_file,
        epoch,
        hist,
        train_loss_iou / len(train_loader),
        train_loss_vec / len(train_loader),
        write=True,
    )
    util.performAngleMetrics(
        train_loss_angle_file, val_loss_angle_file, epoch, hist_angles, write=True
    )

    return train_loss_iou / len(train_loader),p_accu

def valid(epoch):
    global best_accuracy
    global best_miou
    model.eval()
    val_loss_iou = 0
    val_loss_vec = 0
    hist = np.zeros((config["task1_classes"], config["task1_classes"]))
    hist_angles = np.zeros((config["task2_classes"], config["task2_classes"]))
    crop_size = config["val_dataset"][args.dataset]["crop_size"]


    for i, (inputsBGR, labels, vecmap_angles) in enumerate(val_loader, 0):
        with torch.no_grad():
            inputsBGR = Variable(
                inputsBGR.float().cuda(), volatile=True, requires_grad=False
            )

            outputs, pred_vecmaps = model(inputsBGR)
            if args.multi_scale_pred:
                loss1 = road_loss(outputs[0], util.to_variable(labels[0], True, False), True)
                if num_gpus > 1:
                    num_stacks = model.module.num_stacks
                else:
                    if model.num_stacks is None:
                        num_stacks = 2
                    else:
                        num_stacks = model.num_stacks
                for idx in range(num_stacks - 1):
                    loss1 += road_loss(outputs[idx + 1], util.to_variable(labels[0], True, False), True)

                for idx, output in enumerate(outputs[-3:]):
                    if idx <2:
                        loss1 += road_loss(output, util.to_variable(labels[idx + 1]), False)
                    else:
                        loss1 += road_loss(output, util.to_variable(labels[-1]), False)
                loss2 = angle_loss(pred_vecmaps[0], util.to_variable(vecmap_angles[0], True, False))

                for idx in range(num_stacks - 1):
                    loss2 += angle_loss(
                        pred_vecmaps[idx + 1], util.to_variable(vecmap_angles[0], True, False)
                    )
                for idx, pred_vecmap in enumerate(pred_vecmaps[-2:]):
                    loss2 += angle_loss(
                        pred_vecmap, util.to_variable(vecmap_angles[idx + 1], True, False)
                    )

                outputs = outputs[-1]
                pred_vecmaps = pred_vecmaps[-1]
            else:
                outputs = outputs[-1]
                pred_vecmaps = pred_vecmaps[-1]
                loss1 = road_loss(outputs, util.to_variable(labels[0], True, False), True)
                loss2 = angle_loss(pred_vecmaps, util.to_variable(vecmap_angles[0],True, False))

            val_loss_iou += loss1.item()
            val_loss_vec += loss2.item()

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

            p_accu, miou, road_iou, railway_iou, driveway_iou, pathway_iou, bridge_iou, fwacc = util.performMetrics(
                train_loss_file,
                val_loss_file,
                epoch,
                hist,
                val_loss_iou / (i + 1),
                val_loss_vec / (i + 1),
            )

            p_accu_angle, miou_angle, fwacc_angle = util.performAngleMetrics(
                train_loss_angle_file, val_loss_angle_file, epoch, hist_angles, is_train=False
            )

            viz_util.progress_bar(
                i,
                len(val_loader),
                "Loss: %.6f | VecLoss: %.6f | mIoU: %.4f%%(%.4f%%) | rail:%.4f%% | drive:%.4f%% | path:%.4f%% | bridge:%.4f%% | angle miou: %.4f%%"
                % (
                    val_loss_iou / (i + 1),
                    val_loss_vec / (i + 1),
                    miou,
                    road_iou,
                    railway_iou,
                    driveway_iou,
                    pathway_iou,
                    bridge_iou,
                    miou_angle,
                ),
            )

            if i % 100 == 0 or i == len(val_loader) - 1:
                images_path = "{}/images/".format(experiment_dir)
                util.ensure_dir(images_path)
                road_path = "{}/road/".format(experiment_dir)
                util.ensure_dir(road_path)
                angle_path = "{}/angle/".format(experiment_dir)
                util.ensure_dir(angle_path)
                util.savePredictedProb_MTL_5(
                    inputsBGR.data.cpu(),   #输入图像
                    labels[-1].cpu(),       #输入标签
                    predicted.cpu(),        #预测图像
                    predicted_angle.cpu(),
                    os.path.join(images_path, "validate_pair_{}_{}.png".format(epoch, i)),
                    os.path.join(road_path, "road_predict_{}_{}".format(epoch, i)),
                    os.path.join(angle_path, "angle_predict_{}_{}".format(epoch, i)),
                    norm_type=config["val_dataset"]["normalize_type"],
                    crop_size = crop_size
                )

        del inputsBGR, labels, predicted, outputs, pred_vecmaps, predicted_angle

    accuracy, miou, road_iou, railway_iou, driveway_iou, pathway_iou, bridge_iou, fwacc = util.performMetrics(
        train_loss_file,
        val_loss_file,
        epoch,
        hist,
        val_loss_iou / len(val_loader),
        val_loss_vec / len(val_loader),
        is_train=False,
        write=True,
    )
    util.performAngleMetrics(
        train_loss_angle_file,
        val_loss_angle_file,
        epoch,
        hist_angles,
        is_train=False,
        write=True,
    )


    if miou > best_miou:
        best_accuracy = accuracy
        best_miou = miou
        util.save_checkpoint(epoch, val_loss_iou / len(val_loader), model, optimizer, best_accuracy, best_miou, config, experiment_dir)

    return val_loss_iou / len(val_loader),accuracy



for epoch in range(start_epoch, total_epochs + 1):
    start_time = datetime.now()
    scheduler.step(epoch)
    print("\nTraining Epoch: %d" % epoch)
    train_loss,train_acc = train(epoch)
    tb_writer = SummaryWriter('./logs/compare/{}'.format(args.model_name))
    if epoch % config["trainer"]["val_freq"] == 0:
        print("\nValiding Epoch: %d" % epoch)
        val_loss,val_acc = valid(epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)

    tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]

    tb_writer.add_scalar(tags[0], train_loss, epoch)
    tb_writer.add_scalar(tags[1], train_acc, epoch)
    tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
    print('学习率', optimizer.param_groups[0]["lr"])

    end_time = datetime.now()
    print("Time Elapsed for epoch => {1}".format(epoch, end_time - start_time))
