
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
from datasets_.data_loader import load_data
import argparse
from model import DRKL

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

    # 设置消融参数
    # parser.add_argument('--opt_beta', type=bool, default=True, help='消融参数, 是否优化beta, 默认优化')
    # parser.add_argument('--opt_alpha', type=bool, default=True, help='消融参数, 是否优化alpha, 默认优化')
    # parser.add_argument('--opt_feature_net', type=bool, default=True, help='消融参数, 是否优化特征网络, 默认优化')
    parser.add_argument('--opt_beta', type=str2bool, default=False)
    parser.add_argument('--opt_alpha', type=str2bool, default=True)
    parser.add_argument('--opt_feature_net', type=str2bool, default=True)


    # 损失选择
    parser.add_argument('--loss_type', type=str, default='sm', choices=['sm', 'dsm', 'mse'], help='损失函数, score_matching or denoising_score_matching')

    # 设备控制
    parser.add_argument('--device', type=str, default='cuda:4', help='设备, cpu or cuda')
    parser.add_argument('--random_state', type=int, default=None, help='随机种子, 如果为None, 则使用42,0,1')
    parser.add_argument('--save_json', type=str, default=None, help='保存JSON结果文件路径')
    
    # 定义数据集载入参数
    parser.add_argument('--openml_id', type=int, default=42225, help='42712,42821, 42225, 45048, 41540, 42571')
    parser.add_argument('--batch_type', type=str, default='mini', choices=['mini', 'all'], help='训练时, 使用小批量还是全部数据')
    parser.add_argument('--batch_size', type=int, default=512, help='小批量训练时, 每个batch的大小')
    parser.add_argument('--cluster_num', type=int, default=4096, help='聚类数量')    
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--val_size', type=float, default=0.15)

    # 定义核函数参数
    # parser.add_argument('--num_kernels', type=int, default=8, help='核函数数量')
    # parser.add_argument('--sigmas', type=list, default=[1,0.1,1,0.1,1,0.1,1,0.1], help='高斯核参数 带宽, 越大, 核函数越平缓, 取1可得较好结果,列表长度为num_kernels')
    parser.add_argument('--gamma', type=float, default=0.01, help='RKL 最小二乘, 正则项参数 gamma_n, 取5可得较好结果, 调参区间[0.01,0.1,0.5,1,2,5,10,20,50,70,100]')
    
    # 定义特征网络参数
    parser.add_argument('--input_dim', type=int, default=None, help='输入维度')
    parser.add_argument('--feature_dim', type=int, default=20, help='特征维度')
    parser.add_argument('--feature_net_lr', type=float, default=0.001, help='特征网络学习率')
    parser.add_argument('--weight_score_mathching', type=float, default=1.0, help='score matching 损失权重')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='特征网络权重衰减')


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
    # 载入数据
    X_train, X_val, X_test, y_train, y_val, y_test, y_scaler = load_data(openml_id=opt.openml_id, 
                                                                         test_size=opt.test_size, 
                                                                         val_size=opt.val_size, 
                                                                         random_state=opt.random_state)
    # 设置特征网络输入维度
    opt.input_dim = X_train.shape[1]

    # 初始化gamma   缺少按n_kernel的parameter的初始化, 可以考虑设置成在某个数据集大小上给固定值，例如在小于2000数据集，取10, 大于5000取5，大于10000取1
    if opt.gamma is None:
        data_size = X_train.shape[0]
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

            # 消融控制
            opt_beta=opt.opt_beta,
            opt_alpha=opt.opt_alpha,
            opt_feature_net=opt.opt_feature_net,

            # 损失选择
            loss_type=opt.loss_type,

            # 核函数参数
            # num_kernels=opt.num_kernels,
            gamma=opt.gamma,
            # sigmas=opt.sigmas,

            # 特征网络参数
            input_dim=opt.input_dim,
            feature_dim=opt.feature_dim,
            feature_net_lr=opt.feature_net_lr,
            weight_sm=opt.weight_score_mathching,
            weight_decay=opt.weight_decay,

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

            # 消融控制
            opt_beta=opt.opt_beta,
            opt_alpha=opt.opt_alpha,
            opt_feature_net=opt.opt_feature_net,

            # 损失选择
            loss_type=opt.loss_type,

            # 核函数参数    
            # num_kernels=opt.num_kernels,
            gamma=opt.gamma,
            # sigmas=opt.sigmas,

            # 特征网络参数
            input_dim=opt.input_dim,
            feature_dim=opt.feature_dim,
            feature_net_lr=opt.feature_net_lr,
            weight_sm=opt.weight_score_mathching,
            weight_decay=opt.weight_decay,

            # 交替优化参数
            l2_beta_iter=opt.l2_beta_alpha_iter,
            feature_net_iter=opt.feature_net_iter,

            # L2参数
            L_type=opt.L_type,
            l2_beta0=opt.l2_beta0,
            l2_beta_radius=opt.l2_beta_radius,
        )
    # 训练模型
    train_losses, val_losses, test_losses, best_test_mse = model.train(X_train, y_train, 
                                                                        X_val, y_val, 
                                                                        X_test, y_test, 
                                                                        num_epochs=opt.num_epochs, 
                                                                        batch_type=opt.batch_type,
                                                                        batch_size=opt.batch_size,
                                                                        cluster_num=opt.cluster_num,
                                                                        )

    return train_losses, val_losses, test_losses, best_test_mse

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
        train_losses, val_losses, test_losses, best_test_mse = main(opt)
        best_test_mse_list.append(best_test_mse)
    
        result = {
            'openml_id': opt.openml_id,
            'L_type': opt.L_type,
            # 'num_kernels': opt.num_kernels,
            'seed': seed,
            'opt': opt.__dict__,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_losses': test_losses,
            'best_test_mse': best_test_mse
        }
        # 使用绝对路径
        if opt.save_json is None:
            opt.save_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/exp_tmp', f'result_{opt.openml_id}_{opt.L_type}_{seed}_{time.strftime("%Y%m%d_%H%M%S")}.json')
        with open(opt.save_json, 'w') as f:
            json.dump(result, f)
            print(f"JSON结果保存到: {opt.save_json}")
    
    # 打印每个seed的best_test_mse
    for seed, best_test_mse in zip(seed_list, best_test_mse_list):
        print(f'seed: {seed}, best_test_mse: {best_test_mse}')
    # 打印平均值和标准差到屏幕，并保存到result/exp_tmp/文件夹
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/exp_tmp', f'result.csv')
    with open(result_path, 'a') as f:
        f.write(f'{opt.openml_id} {opt.L_type} {np.mean(best_test_mse_list)} {np.std(best_test_mse_list)}\n')
        print(f"结果保存到: {result_path}")
    print(f'{opt.openml_id} {opt.L_type} {np.mean(best_test_mse_list)} {np.std(best_test_mse_list)}')


def run_mode(opt):
    print(opt)
    train_losses, val_losses, test_losses, best_test_mse = main(opt)

    result = {
        'openml_id': opt.openml_id,
        'L_type': opt.L_type,
        'seed': opt.random_state,
        'gamma': opt.gamma,
        'cw': opt.cw,       # for l1
        'l2_beta0': opt.l2_beta0,   # for l2
        'l2_beta_radius': opt.l2_beta_radius,   # for l2
        'opt': opt.__dict__,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_losses': test_losses,
        'best_test_mse': best_test_mse
    }
    # 使用绝对路径
    if opt.save_json is None:
        opt.save_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/exp_tmp', f'result_{opt.openml_id}_{opt.L_type}_{time.strftime("%Y%m%d_%H%M%S")}.json')
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

