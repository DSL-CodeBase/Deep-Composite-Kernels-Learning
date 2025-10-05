"""
在回归任务上使用，包括:
1. 载入数据, 调用data_loader.py
    id_list = [46589, 44956, 44971, 44972, 43918, 44969, 46840]
2. 训练模型
3. 绘制结果



核心问题分析:
1. 预测时，使用多少个点作为支撑进行预测？  全部训练集
2. 训练beta时, 每轮使用全部训练集计算
3. 训练alpha时, 使用全部训练集计算
4. 训练特征网络时, 固定alpha, beta, 采用全部训练集优化特证网络(此时可以选择用batch或不用batch)
5. 预测时,网络需要存储哪些参数? alpha, beta, 特征网络, 支撑点(训练集)通过特征网络得到的特征



那么，优化过程分为两个阶段:
两个阶段交替进行: 
1. 优化alpha, beta:
    (1) 通过特征网络，获取训练集的所有特征
    (2) 使用这些特征计算每个子核的核矩阵
    (3) 对于L1正则化的beta, 使用坐标下降法优化, 每次更新beta, 符合条件后, 更新alpha
    (4) 对于L2正则化的beta, 需要交替优化alpha和beta。设定迭代次数, 先计算alpha, 再计算beta, 交替进行
2. 优化特征网络:
    (1) 固定alpha, beta, 使用全部训练集优化特征网络  (每次反向传播后, 提取的特征就变了, 怎么处理？不进行小批量训练, 只优化一轮？)
    (2) 使用梯度下降法优化

    
架构设计问题补充： 
1. 固定设置4个核: RBF + Polynomial + Laplacian + Wavelet
1.1 RBF参数: sigma = 1, 带宽越大, 核函数越平缓
1.2 Polynomial参数: degree = 2, c = 1多项式次数越高, 核函数越复杂
1.3 Laplacian参数: sigma = 1, 带宽越大, 核函数越平缓
1.4 Wavelet参数: sigma = 1, 带宽越大, 核函数越平缓


这个文件处理大数据集跑不了的问题

"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from datasets_.data_loader_image import load_data
import argparse
from models.image_clf_model_DRKL_last_support import DRKL

import json
import os
import time
import sys


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_opt():
    parser = argparse.ArgumentParser()

    # 运行模式
    parser.add_argument('--mode', type=str, default='debug', help='运行模式, debug or run')
    parser.add_argument('--beizhu', type=str, default=None, help='备注, 用于区分不同实验')

    # 设置消融参数
    # parser.add_argument('--opt_beta', type=bool, default=True, help='消融参数, 是否优化beta, 默认优化')
    # parser.add_argument('--opt_alpha', type=bool, default=True, help='消融参数, 是否优化alpha, 默认优化')
    # parser.add_argument('--opt_feature_net', type=bool, default=True, help='消融参数, 是否优化特征网络, 默认优化')
    parser.add_argument('--opt_beta', type=str2bool, default=False)
    parser.add_argument('--opt_alpha', type=str2bool, default=True)
    parser.add_argument('--opt_feature_net', type=str2bool, default=True)


    # 损失选择
    parser.add_argument('--loss_type', type=str, default='mse', choices=['sm', 'dsm', 'mse'], help='损失函数, score_matching or denoising_score_matching')

    # 设备控制
    parser.add_argument('--device', type=str, default='cuda:3', help='主设备, cpu or cuda')
    parser.add_argument('-A', '--auto_gpu_distribute', type=str2bool, default=True, help='自动将特征网络分布到多个GPU')
    parser.add_argument('--gpu_ids', type=str, default='3', help='指定GPU ID列表，用逗号分隔，如"0,1,2,3"')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子, 如果为None, 则使用42,0,1')
    parser.add_argument('--save_json', type=str, default=None, help='保存JSON结果文件路径')
    
    # 定义数据集载入参数
    parser.add_argument('--dataset_name', type=str, default='mnist', help='数据集名称: cifar10, cifar100, mnist, fashion_mnist, svhn, stl10')
    parser.add_argument('--batch_type', type=str, default='mini', choices=['mini', 'all'], help='训练时, 使用小批量还是全部数据')
    parser.add_argument('--batch_size', type=int, default=256, help='小批量训练时, 每个batch的大小')
    parser.add_argument('--cluster_num', type=int, default=4096, help='聚类数量')    

    # 定义核函数参数
    # parser.add_argument('--num_kernels', type=int, default=8, help='核函数数量')
    # parser.add_argument('--sigmas', type=list, default=[1,0.1,1,0.1,1,0.1,1,0.1], help='高斯核参数 带宽, 越大, 核函数越平缓, 取1可得较好结果,列表长度为num_kernels')
    parser.add_argument('--sigma', type=float, default=1, help='高斯核带宽参数, 越大核函数越平缓, 调参区间[0.01,0.1,0.5,1,2,5,10]')
    parser.add_argument('--gamma', type=float, default=0.001, help='RKL 最小二乘, 正则项参数 gamma_n, 取5可得较好结果, 调参区间[0.01,0.1,0.5,1,2,5,10,20,50,70,100]')
    
    # 定义特征网络参数
    parser.add_argument('--input_channels', type=int, default=3, help='输入图像通道数')
    parser.add_argument('--feature_dim', type=int, default=16, help='特征维度')
    parser.add_argument('--feature_net_lr', type=float, default=0.001, help='特征网络学习率')
    parser.add_argument('--weight_score_mathching', type=float, default=1.0, help='score matching 损失权重')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='特征网络权重衰减')
    parser.add_argument('--pretrained', type=str2bool, default=True, help='是否使用预训练的ResNet权重')


    # 定义beta优化参数_分L1_L2
    parser.add_argument('--L_type', type=str, default='L1', help='beta 正则化类型, L1 or L2')

    # 交替优化参数
    parser.add_argument('--num_epochs', type=int, default=100, help='完整训练轮次')
    parser.add_argument('--l2_beta_alpha_iter', type=int, default=5, help='beta, alpha 交替迭代次数, L2时启用')
    parser.add_argument('--feature_net_iter', type=int, default=1, help='特征网络, alpha 交替迭代次数')

    # L1参数
    parser.add_argument('--cw', type=float, default=10, help='beta L1正则项 系数,调参区间[0.1,0.5,1,2,5,10,20,50,70,100]')
    parser.add_argument('--epsilon', type=float, default=1e-5, help='beta 收敛条件')
    parser.add_argument('--l1_beta_max_iter', type=int, default=100, help='beta 优化最大迭代次数')
    parser.add_argument('--check_interval', type=int, default=10, help='beta 优化检查间隔')
    parser.add_argument('--beta_learning_rate', type=float, default=0.001, help='beta 优化学习率')

    # L2参数
    parser.add_argument('--l2_beta0', type=float, default=0.5, help='beta 圆心, 初始值, 调参区间[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]')
    parser.add_argument('--l2_beta_radius', type=float, default=0.5, help='beta 半径, 调参区间[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]')
    return parser.parse_args()


def main(opt):
    # 设备分配逻辑
    device_list = None
    if opt.auto_gpu_distribute or opt.gpu_ids:
        num_gpus = torch.cuda.device_count()
        print(f"检测到 {num_gpus} 个GPU")
        
        if opt.gpu_ids:
            # 使用指定的GPU ID
            gpu_ids = [int(x.strip()) for x in opt.gpu_ids.split(',') if x.strip()]
            device_list = [f'cuda:{gpu_id}' for gpu_id in gpu_ids]
            print(f"使用指定的GPU: {device_list}")
        elif num_gpus > 1:
            # 自动分配所有可用GPU
            device_list = [f'cuda:{i}' for i in range(num_gpus)]
            print(f"自动分配到所有GPU: {device_list}")
        else:
            print("只有1个GPU，使用单GPU模式")
    
    # 载入图像数据
    train_dataset, test_dataset = load_data(opt.dataset_name)
    
    # 设置输入图像通道数
    if opt.dataset_name in ['mnist', 'fashion_mnist']:
        opt.input_channels = 1
    else:
        opt.input_channels = 3

    # 初始化gamma   缺少按n_kernel的parameter的初始化, 可以考虑设置成在某个数据集大小上给固定值，例如在小于2000数据集，取10, 大于5000取5，大于10000取1
    if opt.gamma is None:
        data_size = len(train_dataset)
        opt.gamma = np.log(data_size) / np.sqrt(data_size)


    # # 设置自动的sigma值，根据特征维度设置合适的范围
    # if opt.sigmas is None:
    #     # 使用特征维度的平方根作为参考，设置sigma的范围
    #     # 较小的sigma用于捕捉局部特征，较大的sigma用于捕捉全局特征
    #     base_sigma = np.sqrt(opt.feature_dim) * 0.1  # 基础sigma值
    #     min_sigma = base_sigma * 0.1  # 最小sigma值
    #     max_sigma = base_sigma * 10.0  # 最大sigma值
        
    #     # 在最小值和最大值之间等比例取num_kernels个值，不包括端点
    #     # 计算等比序列的公比
    #     ratio = (max_sigma / min_sigma) ** (1 / (opt.num_kernels + 1))
    #     opt.sigmas = [min_sigma * (ratio ** (i + 1)) for i in range(opt.num_kernels)]
    #     print(f"自动设置的sigma值: {opt.sigmas}")

    # if len(opt.sigmas) != opt.num_kernels:
    #     opt.sigmas = [1] * opt.num_kernels


    # 定义模型
    if opt.L_type == 'L1':
        model = DRKL(
            #设备控制
            device=opt.device,
            seed=opt.random_state,
            device_list=device_list,

            # 消融控制
            opt_beta=opt.opt_beta,
            opt_alpha=opt.opt_alpha,
            opt_feature_net=opt.opt_feature_net,

            # 损失选择
            loss_type=opt.loss_type,

            # 核函数参数
            # num_kernels=opt.num_kernels,
            sigma=opt.sigma,
            gamma=opt.gamma,
            # sigmas=opt.sigmas,

            # 特征网络参数
            input_channels=opt.input_channels,
            feature_dim=opt.feature_dim,
            feature_net_lr=opt.feature_net_lr,
            weight_sm=opt.weight_score_mathching,
            weight_decay=opt.weight_decay,
            pretrained=opt.pretrained,

            # 交替优化参数
            feature_net_iter=opt.feature_net_iter,

            #L1参数
            L_type=opt.L_type,
            cw=opt.cw,
            epsilon=opt.epsilon,
            l1_beta_max_iter=opt.l1_beta_max_iter,
            check_interval=opt.check_interval,
            beta_learning_rate=opt.beta_learning_rate,
        )
    elif opt.L_type == 'L2':
        model = DRKL(
            #设备控制
            device=opt.device,
            seed=opt.random_state,
            device_list=device_list,

            # 消融控制
            opt_beta=opt.opt_beta,
            opt_alpha=opt.opt_alpha,
            opt_feature_net=opt.opt_feature_net,

            # 损失选择
            loss_type=opt.loss_type,

            # 核函数参数    
            # num_kernels=opt.num_kernels,
            sigma=opt.sigma,
            gamma=opt.gamma,
            # sigmas=opt.sigmas,

            # 特征网络参数
            input_channels=opt.input_channels,
            feature_dim=opt.feature_dim,
            feature_net_lr=opt.feature_net_lr,
            weight_sm=opt.weight_score_mathching,
            weight_decay=opt.weight_decay,
            pretrained=opt.pretrained,

            # 交替优化参数
            l2_beta_iter=opt.l2_beta_alpha_iter,
            feature_net_iter=opt.feature_net_iter,

            # L2参数
            L_type=opt.L_type,
            l2_beta0=opt.l2_beta0,
            l2_beta_radius=opt.l2_beta_radius,
        )
    # 训练模型
    train_losses, test_losses, best_test_acc = model.train(train_dataset, test_dataset, 
                                                            random_state=opt.random_state,
                                                            num_epochs=opt.num_epochs, 
                                                            batch_type=opt.batch_type,
                                                            batch_size=opt.batch_size,
                                                            cluster_num=opt.cluster_num,
                                                            )

    return train_losses, test_losses, best_test_acc

def debug_mode(opt):
    """
    根据opt.random_state, 设置随机种子, 并跑3个seed, 将日志保存至result/exp_tmp/
    """

    if opt.random_state is None:
        seed_list = [42,0,1]
    else:
        seed_list = [opt.random_state]
    best_test_mse_list = []
    for seed in seed_list:
        opt.random_state = seed
        print(opt)
        train_losses, test_losses, best_test_acc = main(opt)
        best_test_mse_list.append(best_test_acc)
    
        result = {
            'dataset_name': opt.dataset_name,
            'L_type': opt.L_type,
            # 'num_kernels': opt.num_kernels,
            'seed': seed,
            'opt': opt.__dict__,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'best_test_acc': best_test_acc
        }
        # 使用绝对路径
        if opt.save_json is None:
            if opt.beizhu is not None:
                opt.save_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/exp_tmp', f'result_{opt.dataset_name}_{opt.L_type}_{opt.beizhu}_{seed}_{time.strftime("%Y%m%d_%H%M%S")}.json')
            else:
                opt.save_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/exp_tmp', f'result_{opt.dataset_name}_{opt.L_type}_{seed}_{time.strftime("%Y%m%d_%H%M%S")}.json')
        with open(opt.save_json, 'w') as f:
            json.dump(result, f)
            print(f"JSON结果保存到: {opt.save_json}")
    
    # 打印每个seed的best_test_acc
    for seed, best_test_acc in zip(seed_list, best_test_mse_list):
        print(f'seed: {seed}, best_test_acc: {best_test_acc}')
    # 打印平均值和标准差到屏幕，并保存到result/exp_tmp/文件夹
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/exp_tmp', f'result.csv')
    with open(result_path, 'a') as f:
        f.write(f'{opt.dataset_name} {opt.L_type} {np.mean(best_test_mse_list)} {np.std(best_test_mse_list)}\n')
        print(f"结果保存到: {result_path}")
    print(f'{opt.dataset_name} {opt.L_type} {np.mean(best_test_mse_list)} {np.std(best_test_mse_list)}')


def run_mode(opt):
    print(opt)
    train_losses, test_losses, best_test_acc = main(opt)

    result = {
        'dataset_name': opt.dataset_name,
        'L_type': opt.L_type,
        'seed': opt.random_state,
        'sigma': opt.sigma,
        'gamma': opt.gamma,
        'cw': opt.cw,       # for l1
        'l2_beta0': opt.l2_beta0,   # for l2
        'l2_beta_radius': opt.l2_beta_radius,   # for l2
        'opt': opt.__dict__,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'best_test_acc': best_test_acc
    }
    # 使用绝对路径
    if opt.save_json is None:
        opt.save_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/exp_tmp', f'result_{opt.dataset_name}_{opt.L_type}_{time.strftime("%Y%m%d_%H%M%S")}.json')
    with open(opt.save_json, 'w') as f:
        json.dump(result, f)
        print(f"JSON结果保存到: {opt.save_json}")

if __name__ == "__main__":
    """
    这个文件支持两种运行方式：
    1. 当外部调用时(批量跑结果时)，传入了随机数种子和关键参数，直接跑并返回就行, 日志根据opt.save_json参数保存
    2. 当直接运行这个文件时(调试), 默认跑3个seed, 将日志保存至result/exp_tmp/

    主要参数：
    1. --mode = 'debug' 或 'run'
    """    
    opt = get_opt()
    if opt.mode == 'debug':
        debug_mode(opt)
    elif opt.mode == 'run':
        run_mode(opt)
    else:
        raise ValueError(f"运行模式错误: {opt.mode}")

