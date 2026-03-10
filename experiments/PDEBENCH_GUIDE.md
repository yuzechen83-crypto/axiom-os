# PDEBench 真实数据集使用指南

## 概述

PDEBench 是一个包含多个偏微分方程基准数据集的开源项目，是替代JHTDB的绝佳选择。

**GitHub**: https://github.com/pdebench/PDEBench

## 可用的数据集

### 1D问题
- `1D_Advection` - 对流方程
- `1D_Burgers` - Burgers方程
- `1D_Diffusion` - 扩散方程
- `1D_CFD` - 可压缩Navier-Stokes
- `1D_ReactionDiffusion` - 反应扩散方程

### 2D问题
- `2D_Darcy` - Darcy流
- `2D_Diffusion` - 扩散方程
- `2D_ShallowWater` - 浅水方程
- `2D_CFD` - 可压缩Navier-Stokes

### 3D问题
- `3D_CFD` - 可压缩Navier-Stokes (最高1024³分辨率)

## 下载方法

### 方法1: 直接下载 (推荐)

```bash
# 安装PDEBench pip包
pip install pdebench

# 使用Python下载
from pdebench.data.download import download_data

# 下载2D Navier-Stokes (约2GB)
download_data(
    '2D_CFD_Rand_M1_0_Eta1e-08_Zeta1e-08',
    './pdebench_data'
)
```

### 方法2: 手动下载

访问 Zenodo 仓库:
- https://doi.org/10.5281/zenodo.6993294

直接下载链接示例:
```
https://zenodo.org/record/6993294/files/2D_CFD_Rand_M1_0_Eta1e-08_Zeta1e-08.hdf5
```

### 方法3: 使用Axiom-OS内置下载器

```python
from axiom_os.datasets.download_pdebench import download_pdebench_dataset

# 下载数据
download_pdebench_dataset(
    dataset_name='2D_CFD',
    save_path='./data',
    resolution='high'  # 'low', 'medium', 'high'
)
```

## 数据格式

PDEBench使用HDF5格式存储:

```python
import h5py

# 加载数据
with h5py.File('2D_CFD_data.hdf5', 'r') as f:
    # 查看可用键
    print(list(f.keys()))
    
    # 通常包含:
    # - 'tensor': 主要数据 (time, x, y, channels)
    # - 'nu': 粘度系数
    # - 'dx', 'dt': 网格参数
    
    data = f['tensor'][:]
    print(f"Data shape: {data.shape}")  # (n_samples, time_steps, nx, ny, channels)
```

## Axiom-OS集成示例

```python
from axiom_os.datasets.pdebench import PDEBenchDataset
from torch.utils.data import DataLoader

# 创建数据集
dataset = PDEBenchDataset(
    data_path='./pdebench_data/2D_CFD_data.hdf5',
    input_steps=4,
    output_steps=1,
    train=True
)

# 创建数据加载器
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 训练FNO-RCLN
for batch in train_loader:
    input_data = batch['input']    # (B, 4, nx, ny, channels)
    target_data = batch['target']  # (B, 1, nx, ny, channels)
    
    # 前向传播
    pred = model(input_data)
    loss = criterion(pred, target_data)
    ...
```

## 推荐的湍流数据集

### 1. 2D Navier-Stokes (推荐作为起点)
- **文件名**: `2D_CFD_Rand_M1_0_Eta1e-08_Zeta1e-08.hdf5`
- **大小**: ~2.3 GB
- **分辨率**: 512×512
- **时间步**: 1000
- **样本数**: 1000

### 2. 3D Navier-Stokes (高级)
- **文件名**: `3D_CFD_Rand_M1_0_Eta1e-08_Zeta1e-08.hdf5`
- **大小**: ~30 GB
- **分辨率**: 128³或256³
- **时间步**: 100
- **样本数**: 100

### 3. 2D Darcy流
- **文件名**: `2D_DarcyFlow_beta1.0_Train.hdf5`
- **大小**: ~500 MB
- **分辨率**: 256×256
- **样本数**: 1000

## 快速开始脚本

```bash
# 1. 创建数据目录
mkdir -p pdebench_data

# 2. 下载2D NS数据 (使用wget)
cd pdebench_data
wget https://zenodo.org/record/6993294/files/2D_CFD_Rand_M1_0_Eta1e-08_Zeta1e-08.hdf5

# 3. 运行Axiom-OS实验
cd ..
python experiments/axiom_pdebench_real.py --data_path ./pdebench_data/2D_CFD_Rand_M1_0_Eta1e-08_Zeta1e-08.hdf5
```

## 数据验证

```python
import h5py
import numpy as np

def validate_pdebench_file(filepath):
    \"\"\"Validate PDEBench data file\"\"\"
    with h5py.File(filepath, 'r') as f:
        print(f"File: {filepath}")
        print(f"Keys: {list(f.keys())}")
        
        if 'tensor' in f:
            data = f['tensor']
            print(f"Tensor shape: {data.shape}")
            print(f"Data range: [{data[:].min():.4f}, {data[:].max():.4f}]")
            
            # Check for NaN/Inf
            if np.any(np.isnan(data[:])):
                print("WARNING: NaN values detected!")
            if np.any(np.isinf(data[:])):
                print("WARNING: Inf values detected!")
                
        if 'nu' in f:
            print(f"Viscosity: {f['nu'][()]}")
            
# 使用示例
validate_pdebench_file('./pdebench_data/2D_CFD_Rand_M1_0_Eta1e-08_Zeta1e-08.hdf5')
```

## 与JHTDB的比较

| 特性 | JHTDB | PDEBench |
|------|-------|----------|
| 数据类型 | 真实DNS (Johns Hopkins) | 合成DNS/LES |
| 访问方式 | 在线API/SciServer | 下载文件 |
| 3D数据 | 1024³ | 128³-256³ |
| 网络依赖 | 需要 | 一次性下载 |
| 数据量 | 无限 (流式) | 固定大小 |
| 真实物理 | 是 (真实实验) | 是 (高保真模拟) |

## 故障排除

### HDF5文件损坏
```bash
# 验证文件完整性
h5dump -H filename.hdf5
```

### 内存不足
```python
# 使用内存映射加载
with h5py.File('large_file.hdf5', 'r') as f:
    data = f['tensor']  # 不加载到内存
    sample = data[0]    # 只加载需要的样本
```

### 下载中断
```bash
# 使用wget断点续传
wget -c https://zenodo.org/record/6993294/files/2D_CFD_data.hdf5
```

## 引用

如果使用PDEBench，请引用:
```
@article{takamoto2022pdebench,
  title={PDEBench: An Extensive Benchmark for Scientific Machine Learning},
  author={Takamoto, Makoto and Praditia, Timothy and Leiteritz, Ralf and 
          MacKinlay, Dan and Alesiani, Francesco and Pflüger, Dirk and 
          Niepert, Mathias},
  journal={arXiv preprint arXiv:2210.07182},
  year={2022}
}
```

## 下一步

1. 下载2D CFD数据 (~2GB)
2. 运行 `axiom_pdebench_real.py` 实验
3. 扩展到3D数据
4. 对比FNO-RCLN与传统CFD方法
