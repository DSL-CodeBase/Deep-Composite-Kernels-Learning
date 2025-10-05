import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

class SingleKernelNet(nn.Module):
    def __init__(self, feature_dim=64, input_dim=54, num_layers=5, activation='relu'):
        super(SingleKernelNet, self).__init__()
        self.feature_dim = feature_dim
        
        layers = []
        layers.extend([
            nn.Linear(input_dim, 100),
            nn.ReLU() if activation == 'relu' else nn.Tanh(),
        ])
        
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(100, 100),
                nn.ReLU() if activation == 'relu' else nn.Tanh(),
            ])
        
        self.features = nn.Sequential(*layers)
        self.feature_map = nn.Sequential(
            nn.Linear(100, feature_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.feature_map(x)
        return x

class MultiKernelNet(nn.Module):
    def __init__(self, input_dim=54, feature_dim=256, num_kernels=4):
        super(MultiKernelNet, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        
        self.kernel_nets = nn.ModuleList([
            SingleKernelNet(input_dim=input_dim, feature_dim=feature_dim)
            for _ in range(num_kernels)
        ])
        
    def forward(self, x):
        features = [net(x) for net in self.kernel_nets]
        return features
    

def gaussian_kernel(x1, x2, sigma=1.0):
    """Compute Gaussian kernel matrix"""
    x1_normalized = F.normalize(x1, p=2, dim=1)
    x2_normalized = F.normalize(x2, p=2, dim=1)
    dist_matrix = torch.cdist(x1_normalized, x2_normalized, p=2)
    return torch.exp(-dist_matrix ** 2 / (2 * sigma ** 2))


def polynomial_kernel(x1, x2, degree=2, c=1):
    """Compute polynomial kernel matrix"""
    x1_normalized = F.normalize(x1, p=2, dim=1)
    x2_normalized = F.normalize(x2, p=2, dim=1)
    return (torch.matmul(x1_normalized, x2_normalized.T) + c) ** degree

def laplacian_kernel(x1, x2, sigma=1.0):
    """Compute Laplacian kernel matrix"""
    x1_normalized = F.normalize(x1, p=2, dim=1)
    x2_normalized = F.normalize(x2, p=2, dim=1)
    dist_matrix = torch.cdist(x1_normalized, x2_normalized, p=2)
    return torch.exp(-dist_matrix / sigma)

def wavelet_kernel(x1, x2, sigma=1.0):
    """Compute wavelet kernel matrix using Morlet wavelet"""
    x1_normalized = F.normalize(x1, p=2, dim=1)
    x2_normalized = F.normalize(x2, p=2, dim=1)
    dist_matrix = torch.cdist(x1_normalized, x2_normalized, p=2)
    cos_term = torch.cos(1.75 * dist_matrix / sigma)
    exp_term = torch.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
    return cos_term * exp_term

def cauchy_kernel(x1, x2, sigma=1.0):
    """Compute Cauchy kernel matrix"""
    x1_normalized = F.normalize(x1, p=2, dim=1)
    x2_normalized = F.normalize(x2, p=2, dim=1)
    return 1.0 / (1.0 + torch.cdist(x1_normalized, x2_normalized, p=2) ** 2 / (sigma ** 2))

def score_matching_loss(x, features_list, beta):
    """Compute Fisher divergence loss for score matching"""
    total_loss = 0
    for i, features in enumerate(features_list):
        features = F.normalize(features, p=2, dim=1)
        
        grad_outputs = torch.ones_like(features)
        first_grads = torch.autograd.grad(
            outputs=features,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        first_grads = torch.clamp(first_grads, min=-10.0, max=10.0)
        
        second_grads = []
        for d in range(x.size(1)):
            second_grad = torch.autograd.grad(
                outputs=first_grads[:, d].sum(),
                inputs=x,
                create_graph=True,
                retain_graph=True,
            )[0][:, d]
            second_grads.append(second_grad)
        
        second_grads = torch.stack(second_grads, dim=1)
        second_grads = torch.clamp(second_grads, min=-10.0, max=10.0)
        
        fisher_divergence = second_grads + 0.5 * first_grads**2
        loss = torch.mean(torch.abs(fisher_divergence))
        loss = loss + 1e-6
        total_loss += loss

    return total_loss


class DRKL:
    def __init__(self, 
                 device='cuda:0',
                 seed=0,
                 opt_beta=True,
                 opt_alpha=True,
                 opt_feature_net=True,
                 loss_type='sm',
                 num_kernels=4, 
                 gamma=1, 
                 input_dim=54, 
                 feature_dim=64, 
                 feature_net_lr=0.001, 
                 weight_sm=1.0, 
                 feature_net_iter=5, 
                 weight_decay=1e-4,
                 L_type='L1',
                 cw=1.0,  
                 epsilon=1e-6, 
                 l1_beta_max_iter=100, 
                 check_interval=10, 
                 beta_learning_rate=0.01,
                 l2_beta0=0.5, 
                 l2_beta_radius=0.5, 
                 l2_beta_iter=5,
                 ):
        """
        Deep Reproducing Kernel Learning (DRKL) model
        """
        self.device = torch.device(device)
        self.seed = seed
        self.opt_beta = opt_beta
        self.opt_alpha = opt_alpha
        self.opt_feature_net = opt_feature_net
        self.loss_type = loss_type
        
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        self.num_kernels = num_kernels
        self.gamma = gamma
        self.kernel_functions = [gaussian_kernel, laplacian_kernel, wavelet_kernel, cauchy_kernel]

        self.feature_net = MultiKernelNet(input_dim=input_dim, feature_dim=feature_dim, num_kernels=num_kernels).to(self.device)
        self.optimizer = optim.Adam(self.feature_net.parameters(), lr=feature_net_lr, weight_decay=weight_decay)
        if self.loss_type == 'mse':
            self.weight_sm = 0.0
        else:
            self.weight_sm = weight_sm

        self.feature_net_iter = feature_net_iter

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

        self.alpha = None

    def train_mini_batch(self, X_train, y_train, X_val, y_val, X_test, y_test, num_epochs, batch_size=512, num_workers=4, cluster_num=512):
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        train_losses = []
        val_losses = []
        test_losses = []

        self.beta = torch.ones((self.num_kernels,),device=self.device) / self.num_kernels
        self.beta.requires_grad_(True)
        self.alpha = None

        for epoch in range(num_epochs):
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                x.requires_grad_(True)

                if batch_idx == 0 and epoch == 0:
                    self.support_feature_list = self.feature_net(x)
                    support_y = y
                else:
                    self.support_feature_list = self.feature_net(x_last)
                    support_y = y_last

                K_list = []
                for i, features in enumerate(self.support_feature_list):
                    kernel_function = self.kernel_functions[i]
                    K_list.append(kernel_function(features, features))

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
                
                if self.opt_feature_net:
                    if self.loss_type == 'dsm':
                        self.fit_feature_net_dsm(x, y, sigma=1.0, sigma_n=0.2)
                    else:
                        train_feature_list = self.feature_net(x)
                        self.fit_feature_net_sm(x, train_feature_list, y)

                x_last = x.detach().clone()
                y_last = y.detach().clone()
                
            with torch.no_grad():
                train_mse = self.eval_mse_batch(train_loader)
                train_losses.append(train_mse)

                val_mse = self.eval_mse_batch(val_loader)
                val_losses.append(val_mse)

                test_mse = self.eval_mse_batch(test_loader)
                test_losses.append(test_mse)

            print(f"Epoch {epoch+1}/{num_epochs}, Train MSE: {train_mse:.6f}, Val MSE: {val_mse:.6f}, Test MSE: {test_mse:.6f}")
            print(f"alpha mean: {self.alpha.mean().item():.6f}, var: {self.alpha.var().item():.6f}, beta: {self.beta}")
            print("-" * 50)

        best_val_mse = min(val_losses)
        best_val_index = val_losses.index(best_val_mse)
        best_test_mse = test_losses[best_val_index]
        print(f"Best test MSE at best validation: {best_test_mse:.6f}")

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
        """Train using all data at once (no mini-batches)"""
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)
        X_train.requires_grad_(True)

        train_losses = []
        val_losses = []
        test_losses = []

        self.train_feature_list = self.feature_net(X_train)
        K_list = []
        for i, features in enumerate(self.train_feature_list):
            kernel_function = self.kernel_functions[i]
            K_list.append(kernel_function(features, features))

        self.beta = torch.ones((self.num_kernels,),device=self.device) / self.num_kernels
        self.beta.requires_grad_(True)
        self.alpha = self.compute_alpha(K_list, y_train)

        for epoch in range(num_epochs):
            for _ in range(self.feature_net_iter):
                if self.opt_feature_net:
                    if self.loss_type == 'dsm':
                        self.fit_feature_net_dsm(X_train, y_train, sigma=1.0, sigma_n=0.2)
                    else:
                        self.fit_feature_net_sm(X_train, self.train_feature_list, y_train)

                self.train_feature_list = self.feature_net(X_train)
                K_list = []
                for i, features in enumerate(self.train_feature_list):
                    kernel_function = self.kernel_functions[i]
                    K_list.append(kernel_function(features, features))

                if self.opt_alpha or epoch<=5:
                    self.alpha = self.compute_alpha(K_list, y_train)

            K_list_copy = [K.detach().clone() for K in K_list]
            if self.L_type == 'L1':
                if self.opt_beta:
                    self.beta = self.fit_l1_beta(K_list_copy, y_train)
                if self.opt_alpha or epoch <= 5:
                    self.alpha = self.compute_alpha(K_list_copy, y_train)
                print(f"alpha mean: {self.alpha.mean().item():.6f}, var: {self.alpha.var().item():.6f}")
                print(f"beta: {self.beta}")

            elif self.L_type == 'L2':
                for _ in range(self.l2_beta_iter):
                    if self.opt_beta:
                        self.beta = self.fit_l2_beta(K_list_copy)
                    if self.opt_alpha or epoch <= 5:
                        self.alpha = self.compute_alpha(K_list_copy, y_train)
                    print(f"alpha mean: {self.alpha.mean().item():.6f}, var: {self.alpha.var().item():.6f}, beta: {self.beta}")

            with torch.no_grad():
                train_pred = self.predict(self.train_feature_list)
                train_mse = torch.mean((train_pred - y_train) ** 2).item()
                train_losses.append(train_mse)

                val_features = self.feature_net(X_val)
                val_pred = self.predict(val_features)
                val_mse = torch.mean((val_pred - y_val) ** 2).item()
                val_losses.append(val_mse)

                test_features = self.feature_net(X_test)
                test_pred = self.predict(test_features)
                test_mse = torch.mean((test_pred - y_test) ** 2).item()
                test_losses.append(test_mse)

            print(f"Epoch {epoch+1}/{num_epochs}, Train MSE: {train_mse:.6f}, Val MSE: {val_mse:.6f}, Test MSE: {test_mse:.6f}")
            print("-" * 50)

        best_val_mse = min(val_losses)
        best_val_index = val_losses.index(best_val_mse)
        best_test_mse = test_losses[best_val_index]
        print(f"Best test MSE at best validation: {best_test_mse:.6f}")
        return train_losses, val_losses, test_losses, best_test_mse


    def train(self, X_train, y_train, X_val, y_val, X_test, y_test, num_epochs, batch_type='mini', batch_size=512, cluster_num=512):
        print(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")
        if batch_type == 'mini':
            train_losses, val_losses, test_losses, best_test_mse = self.train_mini_batch(X_train, y_train, X_val, y_val, X_test, y_test, num_epochs, batch_size=batch_size, cluster_num=cluster_num)
        else:
            train_losses, val_losses, test_losses, best_test_mse = self.train_all_data(X_train, y_train, X_val, y_val, X_test, y_test, num_epochs)
        
        return train_losses, val_losses, test_losses, best_test_mse


    def fit_feature_net_sm(self, x, features_list, labels):
        pred = self.predict(features_list)
        loss_mse = torch.mean((pred - labels) ** 2)
        loss_sm = score_matching_loss(x, features_list, self.beta)
        loss = loss_mse + self.weight_sm * loss_sm

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"Feature network training - loss_mse: {loss_mse:.6f}, loss_sm: {loss_sm:.6f}, loss: {loss:.6f}")
        return loss_mse, loss_sm


    def fit_feature_net_dsm(self, x, y, sigma = 0.1, sigma_n = 0.05):
        """
        Compute denoising score matching loss
        
        Args:
            x: Input data [batch_size, input_dim]
            y: Target values [batch_size]
            sigma: Noise parameter for energy function
            sigma_n: Denoising noise parameter
        
        Returns:
            loss: DSM loss value
        """
        noise = torch.randn_like(x) * sigma_n
        x_tilde = x + noise
        x_tilde.requires_grad_(True)
        
        x_tilde_features = self.feature_net(x_tilde)
        pred = self.predict(x_tilde_features)
        
        energy = (y - pred).pow(2) / (2 * sigma**2)
        
        energy_grad = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=x_tilde,
            create_graph=True,
            retain_graph=True
        )[0]
        
        target = - (x_tilde - x) / (sigma_n**2)
        loss = (energy_grad - target).pow(2).sum(dim=1).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"Feature network training - loss_dsm: {loss:.6f}")

        return loss

    def compute_alpha(self, K_list, y_train):
        K_combined = sum(b * K for b, K in zip(self.beta, K_list))
        K_combined = K_combined + self.gamma * torch.eye(K_combined.size(0), device=K_combined.device)
        alpha = torch.matmul(torch.linalg.inv(K_combined), y_train)
        return alpha


    def fit_l1_beta(self, K_list, y_train):
        """
        Optimize beta parameters using LBFGS optimizer
        Objective: R(β) = y^T K^(-1) y + c_w * ||β||_1
        
        Args:
            K_list: List of kernel matrices
            y_train: Training labels
            
        Returns:
            best_beta: Optimized beta parameters
        """
        beta = self.beta.detach().clone()
        beta.requires_grad_(True)
        
        optimizer = optim.LBFGS(
            [beta],
            lr=self.beta_learning_rate,
            history_size=50,
            line_search_fn='strong_wolfe'
        )
        
        best_loss = float('inf')
        best_beta = beta.data.clone()
        no_improve_count = 0
        
        def compute_objective():
            """Compute objective function value"""
            K_combined = sum(b * K for b, K in zip(beta, K_list))
            n = K_combined.size(0)
            K_reg = K_combined + self.gamma * torch.eye(n, device=K_combined.device)
            
            try:
                L = torch.linalg.cholesky(K_reg)
                alpha = torch.cholesky_solve(y_train, L)
                yt_Kinv_y = torch.matmul(y_train.T, alpha)
            except RuntimeError:
                K_inv = torch.linalg.pinv(K_reg)
                yt_Kinv_y = torch.matmul(y_train.T, torch.matmul(K_inv, y_train))
                
            l1_term = self.cw * torch.sum(torch.abs(beta))
            return yt_Kinv_y + l1_term
        
        def closure():
            """Closure function for LBFGS optimizer"""
            optimizer.zero_grad()
            loss = compute_objective()
            loss.backward()
            return loss
        
        pbar = tqdm(range(self.l1_beta_max_iter), desc="Beta optimization")
        
        for iteration in pbar:
            current_loss = optimizer.step(closure)

            if current_loss < best_loss - self.epsilon:
                best_loss = current_loss
                best_beta = beta.data.clone()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            beta_str = '[' + ', '.join([f'{b:.3f}' for b in beta.data]) + ']'
            
            pbar.set_postfix({
                'loss': f'{current_loss.item():.4f}',
                'best_loss': f'{best_loss.item():.4f}',
                'no_improve': no_improve_count,
                'beta': beta_str,
                'l1_term': f'{self.cw * torch.sum(torch.abs(beta)).item():.3f}'
            })
            
            if no_improve_count >= self.check_interval:
                best_beta_str = '[' + ', '.join([f'{b:.3f}' for b in best_beta]) + ']'
                pbar.set_postfix({
                    'status': 'early_stop',
                    'best_loss': f'{best_loss.item():.4f}',
                    'beta': best_beta_str
                })
                break
        
        pbar.close()
        return best_beta


    def fit_l2_beta(self, K_list):
        """
        Compute beta using L2 constraint
        z_s = alpha.T * K_s * alpha
        beta* = beta0 + beta_radius * z_s / |z|
        """
        z = torch.zeros(self.num_kernels, device=K_list[0].device)
        for s, K in enumerate(K_list):
            z[s] = torch.matmul(self.alpha.T, torch.matmul(K, self.alpha))
        z_norm = torch.norm(z)
        beta_star = self.l2_beta0 + self.l2_beta_radius * z / z_norm
        return beta_star.detach()
    

    def predict(self, features_list):
        """Predict using the trained model"""
        K_test_list = []
        for i, (features, train_features) in enumerate(zip(features_list, self.support_feature_list)):
            kernel_function = self.kernel_functions[i]
            K_test = kernel_function(features, train_features)
            K_test_list.append(K_test)
        
        K_test_combined = sum(b * K for b, K in zip(self.beta, K_test_list))
        pred = torch.matmul(K_test_combined, self.alpha)
        return pred