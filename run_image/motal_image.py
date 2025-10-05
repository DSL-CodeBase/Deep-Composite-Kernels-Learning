import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

class SingleKernelNet(nn.Module):
    def __init__(self, feature_dim=256, input_channels=3, activation='relu', pretrained=False):
        super(SingleKernelNet, self).__init__()
        self.feature_dim = feature_dim
        
        # 使用ResNet18作为backbone
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # 根据输入图像大小调整ResNet结构
        # 对于CIFAR-10等32x32的小图像，需要调整第一层和池化层
        if input_channels != 3:
            # 修改第一层卷积以适应不同的输入通道数
            self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            # 对于32x32图像，使用更小的卷积核和步长
            self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 移除最大池化层以避免对小图像过度下采样
        self.resnet.maxpool = nn.Identity()
        
        # 获取ResNet18最后全连接层的输入维度
        resnet_feature_dim = self.resnet.fc.in_features
        
        # 移除原始的分类层
        self.resnet.fc = nn.Identity()
        
        # 添加自定义的特征映射层
        self.feature_map = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(resnet_feature_dim, feature_dim),
            # nn.ReLU() if activation == 'relu' else nn.Tanh(),
            # nn.Dropout(0.3),
            # nn.Linear(512, feature_dim),
            # nn.BatchNorm1d(feature_dim),  # 添加BatchNorm提高稳定性
        )

    def forward(self, x):
        # x: [batch_size, channels, height, width]
        # 通过ResNet提取特征
        x = self.resnet(x)
        # 通过自定义映射层得到最终特征
        x = self.feature_map(x)
        return x

class MultiKernelNet(nn.Module):
    def __init__(self, input_channels=3, feature_dim=256, num_kernels=4, pretrained=False, device_list=None):
        super(MultiKernelNet, self).__init__()
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.num_kernels = num_kernels
        
        # 获取可用的GPU设备列表
        if device_list is None:
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                # 如果没有GPU，所有网络都在CPU上
                self.device_list = ['cpu'] * num_kernels
            else:
                # 将kernel网络分布到不同GPU上
                self.device_list = [f'cuda:{i % num_gpus}' for i in range(num_kernels)]
        else:
            self.device_list = device_list
            
        print(f"将{num_kernels}个特征网络分布到设备: {self.device_list}")
        
        # 创建多个ResNet18特征网络并分配到不同设备
        self.kernel_nets = nn.ModuleList()
        for i in range(num_kernels):
            net = SingleKernelNet(input_channels=input_channels, feature_dim=feature_dim, pretrained=pretrained)
            net = net.to(self.device_list[i])
            self.kernel_nets.append(net)
        
    def forward(self, x):
        # 获取每个核的特征，处理跨GPU数据传输
        features = []
        
        # 记录输入数据的原始设备
        original_device = x.device
        
        for i, net in enumerate(self.kernel_nets):
            # 将输入数据移动到对应的设备
            x_device = x.to(self.device_list[i])
            # 通过网络获取特征
            feature = net(x_device)
            # 将特征移回原始设备以保持一致性
            feature = feature.to(original_device)
            features.append(feature)
            
        return features
    

def gaussian_kernel(x1, x2, sigma=1.0):
    """计算高斯核矩阵，使用更稳定的实现"""
    # 对特征进行L2归一化
    x1_normalized = F.normalize(x1, p=2, dim=1)
    x2_normalized = F.normalize(x2, p=2, dim=1)
    
    # 计算距离矩阵
    dist_matrix = torch.cdist(x1_normalized, x2_normalized, p=2)
    
    # 使用距离计算核矩阵，添加数值稳定性
    # dist_matrix = torch.clamp(dist_matrix, min=0.0, max=2.0)  # 限制距离范围
    return torch.exp(-dist_matrix ** 2 / (2 * sigma ** 2))

class GaussianKernel:
    def __init__(self, sigma=1.0):
        self.sigma = sigma
    
    def __call__(self, x1, x2):
        return gaussian_kernel(x1, x2, sigma=self.sigma)


def polynomial_kernel(x1, x2, degree=2, c=1):
    # 对特征进行L2归一化
    x1_normalized = F.normalize(x1, p=2, dim=1)
    x2_normalized = F.normalize(x2, p=2, dim=1)

    """计算多项式核矩阵"""
    return (torch.matmul(x1_normalized, x2_normalized.T) + c) ** degree

def laplacian_kernel(x1, x2, sigma=1.0):
    """计算拉普拉斯核矩阵"""
    # 对特征进行L2归一化
    x1_normalized = F.normalize(x1, p=2, dim=1)
    x2_normalized = F.normalize(x2, p=2, dim=1)

    # 计算距离矩阵
    dist_matrix = torch.cdist(x1_normalized, x2_normalized, p=2)
    # 修复：拉普拉斯核的正确公式是 exp(-dist_matrix / sigma)
    return torch.exp(-dist_matrix / sigma)

def wavelet_kernel(x1, x2, sigma=1.0):
    """计算小波核矩阵 - 使用Morlet小波核"""
    # 对特征进行L2归一化
    x1_normalized = F.normalize(x1, p=2, dim=1)
    x2_normalized = F.normalize(x2, p=2, dim=1)

    # 计算距离矩阵
    dist_matrix = torch.cdist(x1_normalized, x2_normalized, p=2)
    
    # 使用Morlet小波核公式
    # K(x,y) = cos(1.75 * ||x-y|| / sigma) * exp(-||x-y||^2 / (2 * sigma^2))
    cos_term = torch.cos(1.75 * dist_matrix / sigma)
    exp_term = torch.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
    return cos_term * exp_term

def cauchy_kernel(x1, x2, sigma=1.0):
    """计算柯西核矩阵"""
    # 对特征进行L2归一化
    x1_normalized = F.normalize(x1, p=2, dim=1)
    x2_normalized = F.normalize(x2, p=2, dim=1)
    
    return 1.0 / (1.0 + torch.cdist(x1_normalized, x2_normalized, p=2) ** 2 / (sigma ** 2))


def score_matching_loss(x, features_list, beta):
    """计算每个核的feature到x的Fisher散度, 确保损失非负"""
    total_loss = 0
    for i, features in enumerate(features_list):
        # features乘以beta, 符合从输出到输入的反向传播关系, 因为beta是输出到输入的权重
        # features = features * beta[i]

        # 对特征进行归一化，增加数值稳定性
        features = F.normalize(features, p=2, dim=1)
        
        # 计算一阶导数 ∂feature/∂x
        grad_outputs = torch.ones_like(features)
        first_grads = torch.autograd.grad(
            outputs=features,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # 对一阶导数进行裁剪，防止梯度爆炸
        first_grads = torch.clamp(first_grads, min=-10.0, max=10.0)
        
        # 计算二阶导数 ∂²feature/∂x²
        second_grads = []
        for d in range(x.size(1)):  # 遍历输入维度
            second_grad = torch.autograd.grad(
                outputs=first_grads[:, d].sum(),
                inputs=x,
                create_graph=True,
                retain_graph=True,
            )[0][:, d]
            second_grads.append(second_grad)
        
        second_grads = torch.stack(second_grads, dim=1)
        # 对二阶导数进行裁剪
        second_grads = torch.clamp(second_grads, min=-10.0, max=10.0)
        
        # 计算Fisher散度损失，使用绝对值确保非负
        fisher_divergence = second_grads + 0.5 * first_grads**2
        loss = torch.mean(torch.abs(fisher_divergence))
        
        # 添加小的常数项确保严格非负
        loss = loss + 1e-6
        
        total_loss += loss

    return total_loss


class DRKL:
    def __init__(self, 
                 # 设备控制
                 device='cuda:0',
                 seed=0,
                 device_list=None,  # 多GPU设备列表，用于分布特征网络

                 # 消融控制
                 opt_beta=True,
                 opt_alpha=True,
                 opt_feature_net=True,
                
                # 损失选择
                 loss_type='sm',

                 # 核函数参数
                #  num_kernels=4, 
                 sigma=1.0,
                 gamma=1, 
                
                 # 特征网络参数
                 input_channels=3, feature_dim=256, feature_net_lr=0.001, weight_sm=1.0, feature_net_iter=5, weight_decay=1e-4,
                 pretrained=False,  # 是否使用预训练的ResNet权重

                 # beta正则化控制
                 L_type='L1',

                 # L1参数
                 cw=1.0,  epsilon=1e-6, l1_beta_max_iter=100, check_interval=10, beta_learning_rate=0.01,

                 # L2参数
                 l2_beta0=0.5, l2_beta_radius=0.5, l2_beta_iter=5,
                 ):
        """
        定义DRKL核函数, 定义特征网络, 评估函数等;
        把核参数的更新过程放到KernelRidgeRegression中; 
        """
        # 设备控制
        self.device = torch.device(device)
        self.seed = seed

        # 消融控制
        self.opt_beta = opt_beta
        self.opt_alpha = opt_alpha
        self.opt_feature_net = opt_feature_net

        # 损失选择
        self.loss_type = loss_type
        
        # 设置随机种子
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        # 核参数初始化
        # self.num_kernels = num_kernels
        self.sigma = sigma
        self.gamma = gamma
        self.kernel_functions = [gaussian_kernel, laplacian_kernel, wavelet_kernel, cauchy_kernel]
        # 创建使用指定sigma的高斯核函数
        # def gaussian_kernel_with_sigma(x1, x2):
        #     return gaussian_kernel(x1, x2, sigma=self.sigma)
        # self.kernel_functions = [gaussian_kernel]
        self.num_kernels = len(self.kernel_functions)


        # 特征网络初始化 - 支持多GPU分布
        self.feature_net = MultiKernelNet(
            input_channels=input_channels, 
            feature_dim=feature_dim, 
            num_kernels=self.num_kernels, 
            pretrained=pretrained,
            device_list=device_list
        )
        # 主要设备仍然是指定的device，用于存储其他参数
        self.main_device = self.device
        self.optimizer = optim.Adam(self.feature_net.parameters(), lr=feature_net_lr, weight_decay=weight_decay)
        if self.loss_type == 'mse':
            self.weight_sm = 0.0
        else:
            self.weight_sm = weight_sm

        # 特征网络迭代次数
        self.feature_net_iter = feature_net_iter

        # beta参数初始化
        self.L_type = L_type
        if self.L_type == 'L1':
            self.cw = cw
            self.epsilon = epsilon
            self.l1_beta_max_iter = l1_beta_max_iter
            self.check_interval = check_interval
            self.beta_learning_rate = beta_learning_rate

        elif self.L_type == 'L2':
            self.l2_beta0 = l2_beta0
            self.l2_beta_radius = l2_beta_radius
            self.l2_beta_iter = l2_beta_iter

        # alpha参数初始化
        self.alpha = None
        
    def print_gpu_memory_usage(self):
        """打印所有GPU的显存使用情况"""
        if hasattr(self.feature_net, 'device_list'):
            print("\n=== GPU显存使用情况 ===")
            for i, device_name in enumerate(self.feature_net.device_list):
                if 'cuda' in device_name:
                    device_id = int(device_name.split(':')[1])
                    allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
                    cached = torch.cuda.memory_reserved(device_id) / 1024**3  # GB
                    print(f"GPU {device_id} (Kernel {i}): 已分配 {allocated:.2f}GB, 缓存 {cached:.2f}GB")
            print("=" * 25)

    def train_mini_batch(self, X_train, y_train, X_val, y_val, X_test, y_test, num_epochs, batch_size=512, num_workers=4, cluster_num=512):
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

        # # 聚类: 对 train_dataset 中的特征进行聚类，只使用特征部分
        # train_features = train_dataset.tensors[0].cpu().numpy()  # 取出特征张量并转换为 numpy
        # cluster_model = KMeans(n_clusters=cluster_num, random_state=self.seed)
        # cluster_labels = cluster_model.fit_predict(train_features)
        # # 获取聚类中心并保存
        # cluster_centers = cluster_model.cluster_centers_
        # self.cluster_centers = torch.tensor(cluster_centers, dtype=torch.float32).to(self.device)

        # # -----------------------------------------------------------
        # # 为每个簇寻找距离聚类中心最近的样本，以便同时保留 X 与 y 信息
        # # -----------------------------------------------------------
        # representative_indices = []
        # for cid in range(cluster_num):
        #     # 找到属于当前簇的样本索引
        #     idx_in_cluster = np.where(cluster_labels == cid)[0]
        #     if idx_in_cluster.size == 0:
        #         # 可能出现空簇，跳过
        #         continue
        #     # 计算这些样本到聚类中心的 L2 距离
        #     cluster_feats = train_features[idx_in_cluster]
        #     center = cluster_centers[cid]
        #     dists = np.linalg.norm(cluster_feats - center, axis=1)
        #     # 选择距离最小的样本
        #     closest_idx = idx_in_cluster[np.argmin(dists)]
        #     representative_indices.append(closest_idx)

        # # 提取代表性样本的 X 和 y
        # rep_X = torch.tensor(X_train[representative_indices], dtype=torch.float32)
        # rep_y = torch.tensor(y_train[representative_indices], dtype=torch.float32)

        # # 保存到实例属性，方便后续使用
        # self.cluster_rep_X = rep_X.to(self.device)
        # self.cluster_rep_y = rep_y.to(self.device)


        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            # pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            # pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            # pin_memory=True
        )

        # 记录训练过程中的损失
        train_losses = []
        val_losses = []
        test_losses = []

        # beta初始化
        self.beta = torch.ones((self.num_kernels,),device=self.device)
        self.beta.requires_grad_(True)
        
        # alpha初始化
        self.alpha = None

        # 直接开始epoch训练
        for epoch in range(num_epochs):
            # 在第一个epoch开始时打印显存使用情况
            if epoch == 0:
                self.print_gpu_memory_usage()
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                x.requires_grad_(True)

                # 使用特征网络获取特征列表，记录在模型里，作为支撑点
                if batch_idx == 0 and epoch == 0:
                    self.support_feature_list = self.feature_net(x)
                    support_y = y
                else:
                    self.support_feature_list = self.feature_net(x_last)
                    support_y = y_last

                # 为每个特征网络使用对应的核函数, 计算子核矩阵
                K_list = []
                for i, features in enumerate(self.support_feature_list):
                    kernel_function = self.kernel_functions[i]
                    K_list.append(kernel_function(features, features))

                """
                优化beta, alpha;
                """
                # 拷贝一个断开计算图的K_list, 用于优化beta
                K_list_copy = [K.detach().clone() for K in K_list]
                if self.opt_alpha or epoch < 1:
                    self.alpha = self.compute_alpha(K_list_copy, support_y)
                    
                if self.L_type == 'L1':
                    if self.opt_beta:
                        self.beta = self.fit_l1_beta(K_list_copy, support_y)
                    if self.opt_alpha or epoch < 1:
                        self.alpha = self.compute_alpha(K_list_copy, support_y)
                    # print(f"alpha均值: {self.alpha.mean().item():.6f}, 方差: {self.alpha.var().item():.6f}")
                    # print(f"beta: {self.beta}")

                elif self.L_type == 'L2':
                    for _ in range(self.l2_beta_iter):
                        if self.opt_beta:
                            self.beta = self.fit_l2_beta(K_list_copy)
                        if self.opt_alpha or epoch < 1:
                            self.alpha = self.compute_alpha(K_list_copy, support_y)
                        # print(f"alpha均值: {self.alpha.mean().item():.6f}, 方差: {self.alpha.var().item():.6f}, beta: {self.beta}")
                
                """
                优化特征网络,alpha;
                """
                if self.opt_feature_net:
                    if self.loss_type == 'dsm':
                        self.fit_feature_net_dsm(x, y, sigma=1.0, sigma_n=0.2)
                    else:
                        train_feature_list = self.feature_net(x)
                        self.fit_feature_net_sm(x, train_feature_list, y)

                # 更新x_last
                x_last = x.detach().clone()
                y_last = y.detach().clone()

            # 每个epoch结束后, 使用优化完的特征网络, 再算一次支撑点特征, 以及beta, alpha; if语句排除消融模式
            # if self.opt_beta and self.opt_alpha and self.opt_feature_net:
            #     train_feature_list = self.feature_net(x)
            #     K_list = []
            #     for i, features in enumerate(train_feature_list):
            #         kernel_function = self.kernel_functions[i]
            #         K_list.append(kernel_function(features, features))
            #     K_list_copy = [K.detach().clone() for K in K_list]
            #     if self.L_type == 'L1':
            #         self.beta = self.fit_l1_beta(K_list_copy, y)
            #         self.alpha = self.compute_alpha(K_list_copy, y)
            #     elif self.L_type == 'L2':
            #         for _ in range(self.l2_beta_iter):
            #             self.beta = self.fit_l2_beta(K_list_copy)
            #             self.alpha = self.compute_alpha(K_list_copy, y)
                
            # 评估当前模型在训练集、验证集和测试集上的性能
            with torch.no_grad():
                # 训练集评估
                train_mse = self.eval_mse_batch(train_loader)
                train_losses.append(train_mse)

                # 验证集评估
                val_mse = self.eval_mse_batch(val_loader)
                val_losses.append(val_mse)

                # 测试集评估
                test_mse = self.eval_mse_batch(test_loader)
                test_losses.append(test_mse)

            # 打印当前轮次的评估结果
            print(f"Epoch {epoch+1}/{num_epochs}, Train MSE: {train_mse:.6f}, Val MSE: {val_mse:.6f}, Test MSE: {test_mse:.6f}")
            print(f"alpha均值: {self.alpha.mean().item():.6f}, 方差: {self.alpha.var().item():.6f}, beta: {self.beta}")
            print("-" * 50)

        # 打印最好的验证集时，测试集的MSE
        best_val_mse = min(val_losses)
        best_val_index = val_losses.index(best_val_mse)
        best_test_mse = test_losses[best_val_index]
        print(f"最好的验证集时，测试集的MSE: {best_test_mse:.6f}")

        return train_losses, val_losses, test_losses, best_test_mse

    def eval_mse_batch(self, data_loader):
        with torch.no_grad():
            mse_list = []
            for batch_idx, (x, y) in enumerate(data_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                features = self.feature_net(x)
                pred = self.predict(features)
                mse = torch.mean((pred - y) ** 2).item()
                mse_list.append(mse)
        return np.mean(mse_list)

    

    def train_all_data(self, X_train, y_train, X_val, y_val, X_test, y_test, num_epochs):
        """
        采用mini-batch训练
        """        
        # 转换数据为PyTorch张量
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)
        X_train.requires_grad_(True)

        # 记录训练过程中的损失
        train_losses = []
        val_losses = []
        test_losses = []

        """
        初始化特征和核矩阵, beta, alpha;
        """
        # 初始化特征和核矩阵        
        self.train_feature_list = self.feature_net(X_train)
        # 为每个特征网络使用对应的核函数
        K_list = []
        for i, features in enumerate(self.train_feature_list):
            kernel_function = self.kernel_functions[i]
            K_list.append(kernel_function(features, features))

        
        # beta初始化
        self.beta = torch.ones((self.num_kernels,),device=self.device)
        self.beta.requires_grad_(True)
        
        # alpha初始化
        self.alpha = self.compute_alpha(K_list, y_train)

        """
        交替优化阶段
        """
        for epoch in range(num_epochs):
            """
            优化特征网络,alpha;
            """
            for _ in range(self.feature_net_iter):
                if self.opt_feature_net:
                    if self.loss_type == 'dsm':
                        self.fit_feature_net_dsm(X_train, y_train, sigma=1.0, sigma_n=0.2)
                    else:
                        self.fit_feature_net_sm(X_train, self.train_feature_list, y_train)

                # 生成新的特征和核矩阵
                self.train_feature_list = self.feature_net(X_train)
                # 为每个特征网络使用对应的核函数
                K_list = []
                for i, features in enumerate(self.train_feature_list):
                    kernel_function = self.kernel_functions[i]
                    K_list.append(kernel_function(features, features))

                if self.opt_alpha or epoch<=5:
                    self.alpha = self.compute_alpha(K_list, y_train)

            """
            优化beta, alpha;
            """
            # 拷贝一个断开计算图的K_list, 用于优化beta
            K_list_copy = [K.detach().clone() for K in K_list]
            if self.L_type == 'L1':
                if self.opt_beta:
                    self.beta = self.fit_l1_beta(K_list_copy, y_train)
                if self.opt_alpha or epoch <= 5:
                    self.alpha = self.compute_alpha(K_list_copy, y_train)
                print(f"alpha均值: {self.alpha.mean().item():.6f}, 方差: {self.alpha.var().item():.6f}")
                print(f"beta: {self.beta}")

            elif self.L_type == 'L2':
                for _ in range(self.l2_beta_iter):
                    if self.opt_beta:
                        self.beta = self.fit_l2_beta(K_list_copy)
                    if self.opt_alpha or epoch <= 5:
                        self.alpha = self.compute_alpha(K_list_copy, y_train)
                    print(f"alpha均值: {self.alpha.mean().item():.6f}, 方差: {self.alpha.var().item():.6f}, beta: {self.beta}")


            # 评估当前模型在训练集、验证集和测试集上的性能
            with torch.no_grad():
                # 训练集评估
                train_pred = self.predict(self.train_feature_list)
                train_mse = torch.mean((train_pred - y_train) ** 2).item()
                train_losses.append(train_mse)

                # 验证集评估
                val_features = self.feature_net(X_val)
                val_pred = self.predict(val_features)
                val_mse = torch.mean((val_pred - y_val) ** 2).item()
                val_losses.append(val_mse)

                # 测试集评估
                test_features = self.feature_net(X_test)
                test_pred = self.predict(test_features)
                test_mse = torch.mean((test_pred - y_test) ** 2).item()
                test_losses.append(test_mse)

            # 打印当前轮次的评估结果
            print(f"Epoch {epoch+1}/{num_epochs}, Train MSE: {train_mse:.6f}, Val MSE: {val_mse:.6f}, Test MSE: {test_mse:.6f}")
            print("-" * 50)

        # 打印最好的验证集时，测试集的MSE
        best_val_mse = min(val_losses)
        best_val_index = val_losses.index(best_val_mse)
        best_test_mse = test_losses[best_val_index]
        print(f"最好的验证集时，测试集的MSE: {best_test_mse:.6f}")
        return train_losses, val_losses, test_losses, best_test_mse


    def train(self, train_dataset, test_dataset, test_size=0.15, random_state=42, num_epochs=100, batch_type='mini', batch_size=512, cluster_num=512):
        # 设置随机种子
        if random_state is not None:
            torch.manual_seed(random_state)
            
        # 打印数据集大小
        print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
        
        # 调用图像训练方法
        if batch_type == 'mini':
            train_losses, test_losses, best_test_acc = self.train_image_mini_batch(
                train_dataset, test_dataset, num_epochs, batch_size, cluster_num)
        else:
            train_losses, test_losses, best_test_acc = self.train_image_all_data(
                train_dataset, test_dataset, num_epochs)

        return train_losses, test_losses, best_test_acc

    def train_image_mini_batch(self, train_dataset, test_dataset, num_epochs, batch_size=512, cluster_num=512):
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # 记录训练过程中的损失
        train_losses = []
        test_losses = []

        # beta初始化
        self.beta = torch.ones((self.num_kernels,), device=self.device)
        self.beta.requires_grad_(True)
        
        # alpha初始化
        self.alpha = None
        
        # 获取类别数量
        num_classes = len(set([label for _, label in test_dataset]))
        print(f"类别数量: {num_classes}")

        # 直接开始epoch训练
        for epoch in range(num_epochs):
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                x.requires_grad_(True)

                # 使用特征网络获取特征列表，记录在模型里，作为支撑点
                if batch_idx == 0 and epoch == 0:
                    self.support_feature_list = self.feature_net(x)
                    support_y = F.one_hot(y, num_classes=num_classes).float()
                else:
                    self.support_feature_list = self.feature_net(x_last)
                    support_y = y_last

                # 为每个特征网络使用对应的核函数, 计算子核矩阵
                K_list = []
                for i, features in enumerate(self.support_feature_list):
                    kernel_function = self.kernel_functions[i]
                    K_list.append(kernel_function(features, features))

                # 优化beta, alpha
                K_list_copy = [K.detach().clone() for K in K_list]
                if self.opt_alpha or epoch < 1:
                    self.alpha = self.compute_alpha(K_list_copy, support_y)
                    
                if self.L_type == 'L1':
                    if self.opt_beta:
                        self.beta = self.fit_l1_beta(K_list_copy, support_y)
                    if self.opt_alpha or epoch < 1:
                        self.alpha = self.compute_alpha(K_list_copy, support_y)

                elif self.L_type == 'L2':
                    for _ in range(self.l2_beta_iter):
                        if self.opt_beta:
                            self.beta = self.fit_l2_beta(K_list_copy)
                        if self.opt_alpha or epoch < 1:
                            self.alpha = self.compute_alpha(K_list_copy, support_y)
                
                # 优化特征网络
                if self.opt_feature_net:
                    if self.loss_type == 'dsm':
                        self.fit_feature_net_dsm(x, F.one_hot(y, num_classes=num_classes).float(), sigma=1.0, sigma_n=0.2)
                    else:
                        train_feature_list = self.feature_net(x)
                        self.fit_feature_net_sm_clf(x, train_feature_list, F.one_hot(y, num_classes=num_classes).float())

                # 更新x_last
                x_last = x.detach().clone()
                y_last = F.one_hot(y, num_classes=num_classes).float()

            # 每个epoch结束后评估模型性能
            with torch.no_grad():
                # 训练集评估
                train_acc = self.eval_accuracy_batch(train_loader, num_classes)
                train_losses.append(train_acc)



                # 测试集评估
                test_acc = self.eval_accuracy_batch(test_loader, num_classes)
                test_losses.append(test_acc)

            # 打印当前轮次的评估结果
            print(f"Epoch {epoch+1}/{num_epochs}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
            if hasattr(self, 'beta'):
                print(f"alpha均值: {self.alpha.mean().item():.6f}, 方差: {self.alpha.var().item():.6f}, beta: {self.beta}")
            print("-" * 50)

        # 返回最后一轮的测试集精度作为最佳精度
        best_test_acc = test_losses[-1]
        print(f"最后一轮测试集精度: {best_test_acc:.4f}")

        return train_losses, test_losses, best_test_acc

    def train_image_all_data(self, train_dataset, test_dataset, num_epochs):
        # 简化版本，暂时返回空结果
        print("全数据训练模式暂未实现，使用mini-batch模式")
        return self.train_image_mini_batch(train_dataset, test_dataset, num_epochs)

    def eval_accuracy_batch(self, data_loader, num_classes):
        with torch.no_grad():
            correct = 0
            total = 0
            for batch_idx, (x, y) in enumerate(data_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                features = self.feature_net(x)
                pred_probs = self.predict_classification(features, num_classes)
                pred = torch.argmax(pred_probs, dim=1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        return correct / total

    def predict_classification(self, features_list, num_classes):
        """用于分类任务的预测方法"""
        K_test_list = []
        # 为每个特征网络使用对应的核函数
        for i, (features, train_features) in enumerate(zip(features_list, self.support_feature_list)):
            kernel_function = self.kernel_functions[i]
            K_test = kernel_function(features, train_features)
            K_test_list.append(K_test)
        
        K_test_combined = sum(b * K for b, K in zip(self.beta, K_test_list))
        pred = torch.matmul(K_test_combined, self.alpha)
        # 应用softmax获取概率分布
        return F.softmax(pred, dim=1)

    def fit_feature_net_sm_clf(self, x, features_list, labels):
        """专门用于分类任务的score matching训练"""
        # 计算分类预测值
        pred = self.predict_classification(features_list, labels.size(1))
        loss_ce = F.cross_entropy(pred, torch.argmax(labels, dim=1))

        # 计算得分匹配损失
        if self.loss_type =="mse":
            loss_sm = 0
        elif self.loss_type == 'sm':
            loss_sm = score_matching_loss(x, features_list, self.beta)

        # 计算总损失
        loss = loss_ce + self.weight_sm * loss_sm

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"特征网络训练 loss_ce: {loss_ce:.6f}, loss_sm: {loss_sm:.6f}, loss: {loss:.6f}")
        return loss_ce, loss_sm

    def fit_feature_net_sm(self, x, features_list, labels):

        # 计算回归预测值
        pred = self.predict(features_list)
        loss_mse = torch.mean((pred - labels) ** 2)

        # 计算得分匹配损失
        # loss_sm = score_matching_loss(x, features_list, self.beta) / self.num_kernels
        if self.loss_type == 'mse':
            loss_sm = 0
        elif self.loss_type == 'sm':
            loss_sm = score_matching_loss(x, features_list, self.beta)


        # 计算总损失
        loss = loss_mse + self.weight_sm * loss_sm

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"特征网络训练 loss_mse: {loss_mse:.6f}, loss_sm: {loss_sm:.6f}, loss: {loss:.6f}")
        return loss_mse, loss_sm


    def fit_feature_net_dsm(self, x, y, sigma = 0.1, sigma_n = 0.05):
        """
        计算去噪得分匹配损失
        
        参数:
        model: 深度核模型
        x: 输入数据 [batch_size, input_dim]
        y: 目标值 [batch_size]
        sigma: 能量函数中的噪声参数
        sigma_n: 去噪噪声参数
        
        返回:
        loss: DSM损失值
        """
        # 1. 添加高斯噪声
        noise = torch.randn_like(x) * sigma_n
        x_tilde = x + noise
        x_tilde.requires_grad_(True)  # 启用梯度计算
        
        # 2. 前向传播获取预测
        x_tilde_features = self.feature_net(x_tilde)
        pred = self.predict(x_tilde_features)
        
        # 3. 计算能量函数 (y - f_v(x))^2/2σ^2
        energy = (y - pred).pow(2) / (2 * sigma**2)
        
        # 4. 计算对数概率梯度 ∇_x log p(x,y)
        # 根据: ∇_x log p(x,y) = ∇_x [-E] = -∇_x E
        energy_grad = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=x_tilde,
            create_graph=True,  # 保留计算图以计算二阶导
            retain_graph=True
        )[0]
        
        # 5. 计算去噪得分匹配目标
        # 目标: - (x_tilde - x)/σ_n^2
        target = - (x_tilde - x) / (sigma_n**2)
        
        # 6. 计算损失: ||∇_x log p + target||^2
        loss = (energy_grad - target).pow(2).sum(dim=1).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"特征网络训练 loss_dsm: {loss:.6f}")

        return loss
    

    # def fit_feature_net_dsm_v2(self, x, y, sigma=0.1, sigma_n=0.2):
    #     """
    #     改进的去噪得分匹配损失
    #     """
    #     # 1. 添加高斯噪声
    #     noise = torch.randn_like(x) * sigma_n
    #     x_tilde = x + noise
    #     x_tilde.requires_grad_(True)
        
    #     # 2. 前向传播获取预测
    #     x_tilde_features = self.feature_net(x_tilde)
    #     pred = self.predict(x_tilde_features)
        
    #     # 3. 使用可学习的缩放参数
    #     scale = torch.exp(self.log_scale) if hasattr(self, 'log_scale') else sigma
    #     energy = (y - pred).pow(2) / (2 * scale**2)
        
    #     # 4. 计算能量梯度
    #     energy_grad = torch.autograd.grad(
    #         outputs=energy.sum(),
    #         inputs=x_tilde,
    #         create_graph=True,
    #         retain_graph=True
    #     )[0]
        
    #     # 5. 计算目标得分 (注意符号正确性)
    #     target = (x - x_tilde) / (sigma_n**2)
        
    #     # 6. 归一化损失计算
    #     diff = energy_grad - target
    #     dim = x.size(1)  # 输入维度
    #     loss_per_sample = torch.sum(diff * diff, dim=1) / dim
    #     loss = loss_per_sample.mean()
        
    #     # 7. 添加正则化项防止缩放参数过大
    #     if hasattr(self, 'log_scale'):
    #         reg_loss = 0.01 * self.log_scale.pow(2)
    #         loss += reg_loss
        
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     print(f"特征网络训练 loss_dsm: {loss.item():.6f}")
    #     return loss

    def compute_alpha(self, K_list, y_train):
        K_combined = sum(b * K for b, K in zip(self.beta, K_list))
        K_combined = K_combined + self.gamma * torch.eye(K_combined.size(0), device=K_combined.device)
        alpha = torch.matmul(torch.linalg.inv(K_combined), y_train)
        return alpha


    def fit_l1_beta(self, K_list, y_train):
        """
        使用LBFGS优化器优化beta参数, 目标函数为:
        R(β) = y^T K^(-1) y + c_w * ||β||_1
        
        Args:
            K_list: 核矩阵列表
            y_train: 训练标签
            
        Returns:
            best_beta: 优化后的beta参数
        """
        # 创建可优化的beta参数
        beta = self.beta.detach().clone()
        beta.requires_grad_(True)
        
        # 初始化LBFGS优化器
        optimizer = optim.LBFGS(
            [beta],
            lr=self.beta_learning_rate,
            history_size=50,
            line_search_fn='strong_wolfe'
        )
        
        # 记录优化过程
        best_loss = float('inf')
        best_beta = beta.data.clone()
        no_improve_count = 0
        
        def compute_objective():
            """计算目标函数值"""
            # 计算组合核矩阵
            K_combined = sum(b * K for b, K in zip(beta, K_list))
            
            # 添加正则化项
            n = K_combined.size(0)
            K_reg = K_combined + self.gamma * torch.eye(n, device=K_combined.device)
            
            # 计算 y^T K^(-1) y
            try:
                # 使用Cholesky分解求解
                L = torch.linalg.cholesky(K_reg)
                alpha = torch.cholesky_solve(y_train, L)
                # yt_Kinv_y = torch.sum(y_train * alpha)
                yt_Kinv_y = torch.matmul(y_train.T, alpha)
            except RuntimeError:
                # 如果Cholesky分解失败，使用伪逆
                K_inv = torch.linalg.pinv(K_reg)
                # yt_Kinv_y = torch.sum(y_train * torch.matmul(K_inv, y_train))
                yt_Kinv_y = torch.matmul(y_train.T, torch.matmul(K_inv, y_train))
                
            # 计算L1正则项
            l1_term = self.cw * torch.sum(torch.abs(beta))
            
            return yt_Kinv_y + l1_term
        
        def closure():
            """LBFGS优化器的闭包函数"""
            optimizer.zero_grad()
            loss = compute_objective()
            loss.backward()
            return loss
        
        # 创建进度条
        pbar = tqdm(range(self.l1_beta_max_iter), desc="Beta优化进度")
        
        # 优化过程
        for iteration in pbar:
            current_loss = optimizer.step(closure)
            # # 投影：确保 beta ≥ 0，必要时可同时归一化到和为 1
            # with torch.no_grad():
            #     beta.data.clamp_(min=0.0)
            #     # 若想保持和为 1，可取消注释下一行
            #     if beta.sum() > 0: beta.data /= beta.sum()

            # 检查是否找到更好的解
            if current_loss < best_loss - self.epsilon:
                best_loss = current_loss
                best_beta = beta.data.clone()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # 将beta转换为字符串形式
            beta_str = '[' + ', '.join([f'{b:.3f}' for b in beta.data]) + ']'
            
            # 更新进度条描述
            pbar.set_postfix({
                'loss': f'{current_loss.item():.4f}',
                'best_loss': f'{best_loss.item():.4f}',
                'no_improve': no_improve_count,
                'beta': beta_str,
                'l1_term': f'{self.cw * torch.sum(torch.abs(beta)).item():.3f}'
            })
            
            # 早停检查
            if no_improve_count >= self.check_interval:  # 连续5次没有改善就停止
                best_beta_str = '[' + ', '.join([f'{b:.3f}' for b in best_beta]) + ']'
                pbar.set_postfix({
                    'status': 'early_stop',
                    'best_loss': f'{best_loss.item():.4f}',
                    'beta': best_beta_str
                })
                break
        
        # 关闭进度条
        pbar.close()                
        
        # 确保 beta 非负且和为 1（必要时可关闭此步骤以保留原始值）：
        # best_beta = torch.clamp(best_beta, min=0.0)
        # if best_beta.sum() > 0:
        #     best_beta = best_beta / best_beta.sum()
 
        return best_beta


    def fit_l2_beta(self, K_list):
        """
        z_s = alpha.T * K_s * alpha
        beta* = beta0 + beta_radius * z_s / |z|
        """
        # 计算z_s
        z = torch.zeros(self.num_kernels, device=K_list[0].device)
        for s, K in enumerate(K_list):
            z[s] = torch.matmul(self.alpha.T, torch.matmul(K, self.alpha))
        # 计算|z|
        z_norm = torch.norm(z)
        # 计算beta*
        beta_star = self.l2_beta0 + self.l2_beta_radius * z / z_norm
        return beta_star.detach()
    

    def predict(self, features_list):
        """使用训练好的模型进行预测"""
        K_test_list = []
        # 为每个特征网络使用对应的核函数
        for i, (features, train_features) in enumerate(zip(features_list, self.support_feature_list)):
            kernel_function = self.kernel_functions[i]
            K_test = kernel_function(features, train_features)
            K_test_list.append(K_test)
        
        K_test_combined = sum(b * K for b, K in zip(self.beta, K_test_list))
        pred = torch.matmul(K_test_combined, self.alpha)
        return pred