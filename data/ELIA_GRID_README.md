# Elia Grid Data - 数据获取指南

## Operation "Grid Pulse" - 比利时电网频率与不平衡量

### 数据来源

- **Elia Open Data - Frequency** (ods057): https://opendata.elia.be/explore/dataset/ods057
- **Elia Open Data - System Imbalance** (ods126): https://opendata.elia.be/explore/dataset/ods126

### 自动下载（推荐）

```bash
# 从 Elia API 拉取真实数据（需可访问 opendata.elia.be）
python -m axiom_os.scripts.download_elia_real

# 网络不可达时，生成样本格式 CSV
python -m axiom_os.scripts.download_elia_real --sample
```

输出：`axiom_os/data/elia_grid.csv`，`load_elia_grid()` 会自动识别。

### 手动下载步骤

1. 访问 Elia Open Data 网站
2. 选择 **Frequency** 数据，导出 2023 或 2024 年 CSV
3. 选择 **System Imbalance** 数据，导出对应时间段 CSV
4. 根据 Timestamp 将两个表 merge
5. 将合并后的 CSV 保存为 `elia_2022.csv` 或 `elia_2023.csv`，放入本目录

### 预期 CSV 列

| 列名 | 说明 | 示例 |
|------|------|------|
| timestamp / datetime | 时间戳 | 2022-01-01 00:00:00 |
| frequency_Hz | 频率 (Hz) | 50.012 |
| imbalance_MW | 系统不平衡 (MW) | 125.3 |

列名可灵活匹配（含 "freq"/"hz"/"imbal"/"regul"/"mw" 等关键词即可）。

### 合成数据（无真实数据时）

若未提供 CSV，`load_elia_grid(use_synthetic_if_fail=True)` 将自动生成逼真合成数据用于测试。

### 运行

```bash
python main_grid.py
# 或
python -m axiom_os.validate_all --grid-pulse
```
