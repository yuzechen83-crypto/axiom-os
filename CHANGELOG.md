# Changelog

## [Unreleased]

### Added

- **JHTDB LES-SGS 湍流建模**
  - TBNN (Tensor Basis Neural Network): Pope 不变量 + 张量基归一化
  - FNO3d: 3D Fourier Neural Operator 非局部映射
  - 真实 DNS 数据 (Johns Hopkins Turbulence Database)
  - 空间划分严格评估 (train z>4, test z≤4)
  - 3D 可视化: `jhtdb_les_sgs_3d.png`, `jhtdb_les_sgs_comparison.png`

- **公式结晶**
  - `crystallize_formulas.py`: RAR 符号公式提取
  - 保存至 `crystallized_formulas.json`
  - `--to-hippocampus` 可选结晶到知识库

- **Layers**
  - `turbulence_invariants.py`: Pope 不变量、张量基、σ 归一化
  - `tbnn.py`: TBNN 模型
  - `pinn_lstm.py`: PhysicsInformedLSTM
  - `fno.py`: SpectralConv3d, FNO3d

- **Tests**
  - `test_jhtdb_strict.py`: TBNN 严格空间划分
  - `test_jhtdb_fno.py`: FNO3d 训练与评估

- **validate_all**
  - `--jhtdb-les-sgs`: JHTDB 湍流实验
  - `--crystallize`: 公式结晶

### Changed

- TBNN 训练: 增加 epochs (1000)、LR 调度、hidden=48
- README: 补充 JHTDB、公式结晶、湍流建模章节
- 本地部署.md: 补充 JHTDB、crystallize 命令
- GitHub Pages: 首页增加 JHTDB、结晶结果链接
