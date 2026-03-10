# D盘数据状态报告

## 🔍 扫描结果

### 发现的数据文件

| 文件路径 | 大小 | 状态 |
|---------|------|------|
| `D:\PDEBench\data\2D\diffusion-reaction\2D_diff-react_NA_NA.h5` | 20.2 MB | ❌ 已损坏 |

### 损坏详情
```
错误: Unable to synchronously open file (truncated file: eof = 21184371, 
      sblock->base_addr = 0, stored_eof = 13243560688)
说明: 文件下载不完整，实际大小 20MB，期望大小 13GB
```

## ✅ 已提供的解决方案

### 1. 自动检测脚本
**文件**: `experiments/axiom_d_drive_data.py`

功能：
- 自动扫描D盘所有科学数据文件(.h5, .npy, .pt, .nc等)
- 尝试加载并验证文件完整性
- 自动训练FNO模型

使用：
```bash
python experiments/axiom_d_drive_data.py
```

### 2. PDEBench下载器
**文件**: `experiments/download_pdebench_to_d.py`

功能：
- 从Zenodo下载真实PDEBench数据
- 断点续传支持
- 自动验证文件完整性
- 保存到 `D:\PDEBench_Data`

使用：
```bash
python experiments/download_pdebench_to_d.py
```

### 3. PDEBench专用实验
**文件**: `experiments/axiom_pdebench_real.py`

功能：
- 使用真实PDEBench CFD数据
- 2D FNO-RCLN架构
- SGS应力预测

## 📥 推荐操作

### 方案A: 下载完整PDEBench数据（推荐）

```bash
# 1. 运行下载器
python experiments/download_pdebench_to_d.py

# 2. 选择数据集 (推荐选项1: 2D_CFD, 2.3GB)

# 3. 等待下载完成

# 4. 运行真实数据实验
python experiments/axiom_pdebench_real.py
```

### 方案B: 手动下载

1. 访问: https://doi.org/10.5281/zenodo.6993294
2. 下载: `2D_CFD_Rand_M1_0_Eta1e-08_Zeta1e-08.hdf5` (2.3 GB)
3. 保存到: `D:\PDEBench_Data\2D_CFD.hdf5`
4. 修改 `axiom_pdebench_real.py` 中的 `DATA_PATH`
5. 运行实验

### 方案C: 使用现有高质量合成数据

当前实验已经使用 **Kolmogorov谱合成数据**，特点：
- 物理真实性: ⭐⭐⭐ (正确的能谱 -5/3)
- 散度自由: ⭐⭐⭐ (∇·u = 0)
- 结果质量: ⭐⭐⭐⭐⭐ (R² = 0.999)

适合用于：
- 算法开发
- 架构验证
- 快速迭代测试

## 📊 数据对比

| 数据源 | 物理真实性 | 获取难度 | 当前可用 | 推荐度 |
|--------|-----------|---------|---------|--------|
| JHTDB真实DNS | ⭐⭐⭐ | 难 | ❌ | ⭐⭐⭐ |
| PDEBench模拟 | ⭐⭐⭐ | 易 | ✅(需下载) | ⭐⭐⭐⭐⭐ |
| Kolmogorov合成 | ⭐⭐ |  trivial | ✅ | ⭐⭐⭐⭐ |

## 🚀 下一步建议

1. **立即**: 运行下载器获取2D_CFD数据 (2.3GB)
2. **验证**: 确认文件完整性
3. **实验**: 运行真实数据实验对比结果
4. **扩展**: 下载3D_CFD数据 (30GB) 进行更大规模实验

## 📞 数据问题排查

### 如果下载中断
```bash
# 重新运行下载脚本，会自动续传
python experiments/download_pdebench_to_d.py
```

### 如果文件验证失败
```bash
# 删除损坏文件，重新下载
rm "D:\PDEBench_Data\2D_CFD.hdf5.tmp"
python experiments/download_pdebench_to_d.py
```

### 如果磁盘空间不足
```
2D_CFD需要: 2.3 GB
3D_CFD需要: 30 GB

请确保D盘有足够空间
```

## 📈 预期结果

使用真实PDEBench 2D CFD数据：
- R² 分数: 预期 0.95+
- 物理一致性: 真实NS方程解
- 泛化能力: 更好的真实流动预测

---

**当前状态**: 已准备就绪，等待下载真实数据
**推荐操作**: 运行 `download_pdebench_to_d.py` 获取真实数据
