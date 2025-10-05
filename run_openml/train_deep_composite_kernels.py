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
from data_loader import load_data
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

    parser.add_argument('--mode', type=str, default='debug', help='Running mode: debug or run')
    parser.add_argument('--opt_beta', type=str2bool, default=False)
    parser.add_argument('--opt_alpha', type=str2bool, default=True)
    parser.add_argument('--opt_feature_net', type=str2bool, default=True)
    parser.add_argument('--loss_type', type=str, default='sm', choices=['sm', 'dsm', 'mse'], help='Loss function type')
    parser.add_argument('--device', type=str, default='cuda:4', help='Device: cpu or cuda')
    parser.add_argument('--random_state', type=int, default=None, help='Random seed, if None, use [42,0,1]')
    parser.add_argument('--save_json', type=str, default=None, help='Path to save JSON results')
    parser.add_argument('--openml_id', type=int, default=42225, help='OpenML dataset ID')
    parser.add_argument('--batch_type', type=str, default='mini', choices=['mini', 'all'], help='Batch type: mini-batch or all data')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for mini-batch training')
    parser.add_argument('--cluster_num', type=int, default=4096, help='Number of clusters')    
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--val_size', type=float, default=0.15)
    parser.add_argument('--gamma', type=float, default=0.01, help='Regularization parameter gamma_n')
    parser.add_argument('--input_dim', type=int, default=None, help='Input dimension')
    parser.add_argument('--feature_dim', type=int, default=20, help='Feature dimension')
    parser.add_argument('--feature_net_lr', type=float, default=0.001, help='Feature network learning rate')
    parser.add_argument('--weight_score_mathching', type=float, default=1.0, help='Score matching loss weight')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for feature network')
    parser.add_argument('--L_type', type=str, default='L1', help='Regularization type: L1 or L2')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--l2_beta_alpha_iter', type=int, default=5, help='Alternating iterations for beta and alpha (L2 only)')
    parser.add_argument('--feature_net_iter', type=int, default=1, help='Feature network iterations')
    parser.add_argument('--cw', type=float, default=10, help='L1 regularization coefficient for beta')
    parser.add_argument('--epsilon', type=float, default=1e-5, help='Convergence threshold for beta')
    parser.add_argument('--l1_beta_max_iter', type=int, default=100, help='Maximum iterations for beta optimization')
    parser.add_argument('--check_interval', type=int, default=10, help='Check interval for beta optimization')
    parser.add_argument('--beta_learning_rate', type=float, default=0.001, help='Learning rate for beta optimization')
    parser.add_argument('--l2_beta0', type=float, default=0.5, help='Initial value for beta (L2 center)')
    parser.add_argument('--l2_beta_radius', type=float, default=0.5, help='Radius for beta (L2)')
    return parser.parse_args()


def main(opt):
    X_train, X_val, X_test, y_train, y_val, y_test, y_scaler = load_data(openml_id=opt.openml_id, 
                                                                         test_size=opt.test_size, 
                                                                         val_size=opt.val_size, 
                                                                         random_state=opt.random_state)
    opt.input_dim = X_train.shape[1]

    if opt.gamma is None:
        data_size = X_train.shape[0]
        opt.gamma = np.log(data_size) / np.sqrt(data_size)

    if opt.L_type == 'L1':
        model = DRKL(
            device=opt.device,
            seed=opt.random_state,
            opt_beta=opt.opt_beta,
            opt_alpha=opt.opt_alpha,
            opt_feature_net=opt.opt_feature_net,
            loss_type=opt.loss_type,
            gamma=opt.gamma,
            input_dim=opt.input_dim,
            feature_dim=opt.feature_dim,
            feature_net_lr=opt.feature_net_lr,
            weight_sm=opt.weight_score_mathching,
            weight_decay=opt.weight_decay,
            feature_net_iter=opt.feature_net_iter,
            L_type=opt.L_type,
            cw=opt.cw,
            epsilon=opt.epsilon,
            l1_beta_max_iter=opt.l1_beta_max_iter,
            check_interval=opt.check_interval,
            beta_learning_rate=opt.beta_learning_rate,
        )
    elif opt.L_type == 'L2':
        model = DRKL(
            device=opt.device,
            seed=opt.random_state,
            opt_beta=opt.opt_beta,
            opt_alpha=opt.opt_alpha,
            opt_feature_net=opt.opt_feature_net,
            loss_type=opt.loss_type,
            gamma=opt.gamma,
            input_dim=opt.input_dim,
            feature_dim=opt.feature_dim,
            feature_net_lr=opt.feature_net_lr,
            weight_sm=opt.weight_score_mathching,
            weight_decay=opt.weight_decay,
            l2_beta_iter=opt.l2_beta_alpha_iter,
            feature_net_iter=opt.feature_net_iter,
            L_type=opt.L_type,
            l2_beta0=opt.l2_beta0,
            l2_beta_radius=opt.l2_beta_radius,
        )
    
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
    Run experiments with multiple seeds and save results to results/exp_tmp/
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
            'seed': seed,
            'opt': opt.__dict__,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_losses': test_losses,
            'best_test_mse': best_test_mse
        }
        
        if opt.save_json is None:
            opt.save_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/exp_tmp', f'result_{opt.openml_id}_{opt.L_type}_{seed}_{time.strftime("%Y%m%d_%H%M%S")}.json')
        with open(opt.save_json, 'w') as f:
            json.dump(result, f)
            print(f"JSON results saved to: {opt.save_json}")
    
    for seed, best_test_mse in zip(seed_list, best_test_mse_list):
        print(f'seed: {seed}, best_test_mse: {best_test_mse}')
    
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/exp_tmp', f'result.csv')
    with open(result_path, 'a') as f:
        f.write(f'{opt.openml_id} {opt.L_type} {np.mean(best_test_mse_list)} {np.std(best_test_mse_list)}\n')
        print(f"Results saved to: {result_path}")
    print(f'{opt.openml_id} {opt.L_type} {np.mean(best_test_mse_list)} {np.std(best_test_mse_list)}')


def run_mode(opt):
    print(opt)
    train_losses, val_losses, test_losses, best_test_mse = main(opt)

    result = {
        'openml_id': opt.openml_id,
        'L_type': opt.L_type,
        'seed': opt.random_state,
        'gamma': opt.gamma,
        'cw': opt.cw,
        'l2_beta0': opt.l2_beta0,
        'l2_beta_radius': opt.l2_beta_radius,
        'opt': opt.__dict__,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_losses': test_losses,
        'best_test_mse': best_test_mse
    }
    
    if opt.save_json is None:
        opt.save_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/exp_tmp', f'result_{opt.openml_id}_{opt.L_type}_{time.strftime("%Y%m%d_%H%M%S")}.json')
    with open(opt.save_json, 'w') as f:
        json.dump(result, f)
        print(f"JSON results saved to: {opt.save_json}")

if __name__ == "__main__":
    """
    This script supports two running modes:
    1. 'run' mode: Run with specified parameters and random seed
    2. 'debug' mode: Run with multiple seeds (default: [42,0,1]) for testing
    
    Main parameter:
    --mode: 'debug' or 'run'
    """    
    opt = get_opt()
    if opt.mode == 'debug':
        debug_mode(opt)
    elif opt.mode == 'run':
        run_mode(opt)
    else:
        raise ValueError(f"Invalid mode: {opt.mode}")

