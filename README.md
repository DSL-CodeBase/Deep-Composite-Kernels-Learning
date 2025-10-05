# Deep Composite Kernels Learning via Regularized Fisher Discrepancy

## Project Overview

This project implements a Deep Regularized Kernel Learning (DRKL) method based on Regularized Fisher Discrepancy. The method combines deep neural networks with kernel methods to achieve excellent performance on both regression and classification tasks.

## Key Features

- **Deep Composite Kernel Learning**: Combines multiple kernel functions (Gaussian, Laplacian, Wavelet, Cauchy) for composite learning
- **Fisher Discrepancy Regularization**: Uses Fisher discrepancy as a regularization term to improve model generalization
- **Alternating Optimization Strategy**: Employs alternating optimization of feature networks, kernel weights, and regression coefficients
- **Multi-task Support**: Supports both regression and classification tasks
- **Multi-GPU Support**: Supports multi-GPU parallel training for improved efficiency

## Project Structure

```
├── run_openml/                    # OpenML dataset experiments
│   ├── train_deep_composite_kernels.py  # Main training script
│   ├── model.py                   # DRKL model implementation
│   └── data_loader.py             # Data loader
├── run_image/                     # Image dataset experiments
│   ├── train_image_composite_kernels.py  # Image training script
│   ├── motal_image.py             # Image model implementation
│   └── data_loader_image.py      # Image data loader
└── README.md                      # Project documentation
```

## Core Algorithms

### 1. Deep Composite Kernel Model

The model uses multiple deep feature networks to extract features, then computes similarity through different kernel functions:

```python
# Kernel function combination
kernel_functions = [gaussian_kernel, laplacian_kernel, wavelet_kernel, cauchy_kernel]

# Composite kernel matrix
K_combined = sum(beta[i] * K_i for i, K_i in enumerate(K_list))
```

### 2. Fisher Discrepancy Regularization

Uses Fisher discrepancy as a regularization term to improve model generalization:

```python
def score_matching_loss(x, features_list, beta):
    # Compute first and second order derivatives
    first_grads = torch.autograd.grad(outputs=features, inputs=x, ...)
    second_grads = torch.autograd.grad(outputs=first_grads, inputs=x, ...)
    
    # Fisher discrepancy loss
    fisher_divergence = second_grads + 0.5 * first_grads**2
    return torch.mean(torch.abs(fisher_divergence))
```

### 3. Alternating Optimization Strategy

Employs three-stage alternating optimization:

1. **Feature Network Optimization**: Fix kernel weights β and regression coefficients α, optimize feature network parameters
2. **Kernel Weight Optimization**: Fix feature network and α, optimize kernel weights β (supports L1 and L2 regularization)
3. **Regression Coefficient Optimization**: Fix feature network and β, compute optimal regression coefficients α

## Installation

```bash
pip install torch torchvision
pip install scikit-learn
pip install openml
pip install pandas numpy
pip install matplotlib tqdm
```

## Usage

### 1. OpenML Dataset Experiments

```bash
cd run_openml
python train_deep_composite_kernels.py --mode debug --openml_id 42225 --L_type L1
```

Main parameters:
- `--mode`: Running mode (debug/run)
- `--openml_id`: OpenML dataset ID
- `--L_type`: Regularization type (L1/L2)
- `--device`: Computing device
- `--num_epochs`: Number of training epochs

### 2. Image Dataset Experiments

```bash
cd run_image
python train_image_composite_kernels.py --mode debug --dataset_name cifar10 --L_type L1
```

Main parameters:
- `--dataset_name`: Dataset name (cifar10, mnist, fashion_mnist, etc.)
- `--pretrained`: Whether to use pretrained weights
- `--auto_gpu_distribute`: Whether to automatically distribute GPUs


## Model Parameters


### Feature Network Parameters
- `feature_dim`: Feature dimension
- `feature_net_lr`: Feature network learning rate
- `weight_decay`: Weight decay
- `pretrained`: Whether to use pretrained weights

### Optimization Parameters
- `L_type`: Regularization type (L1/L2)
- `cw`: L1 regularization coefficient
- `l2_beta0`: L2 regularization center
- `l2_beta_radius`: L2 regularization radius

## Experimental Results

The model achieves excellent performance on multiple datasets:

- **Regression Tasks**: Significant improvements over traditional kernel methods on OpenML datasets
- **Classification Tasks**: Achieves or exceeds performance of existing methods on image classification tasks
- **Computational Efficiency**: Multi-GPU support significantly improves training speed

## Technical Features

1. **Numerical Stability**: Uses L2 normalization and gradient clipping to ensure training stability
2. **Memory Optimization**: Supports mini-batch training, adapting to large-scale datasets
3. **Extensibility**: Modular design, easy to extend with new kernel functions and loss functions
4. **Experiment-friendly**: Supports multiple running modes and parameter configurations

## Citation

If you use this project, please cite the related paper:

```bibtex
@article{deep_composite_kernels,
  title={Deep Composite Kernels Learning via Regularized Fisher Discrepancy},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or suggestions, please contact:
- Email: your.email@example.com
- GitHub Issues: [Project Issues Page]

## Changelog

- **v1.0.0**: Initial version with basic deep composite kernel learning
- **v1.1.0**: Added multi-GPU support and image dataset experiments
- **v1.2.0**: Optimized numerical stability and training efficiency