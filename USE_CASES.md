# Axiom-OS 使用场景扩展

## 当前支持的场景

| 场景 | 数据类型 | 状态 |
|------|----------|------|
| 电网频率 | Elia Grid (比利时) | ✅ 可用 |
| 湍流建模 | JHTDB (约翰霍普金斯) | ✅ 可用 |
| 电池RUL | NASA 锂电池 | ✅ 可用 |
| 公式发现 | 合成数据 | ✅ 可用 |

---

## 🚀 扩展场景计划

### 1. 气象与气候
- **温度预测**: 全球气象站数据
- **风速预测**: 风电场数据
- **降雨量**: 区域降水数据
- **数据源**: NOAA, Open-Meteo, 气象局

### 2. 金融量化
- **股票预测**: 历史价格、成交量
- **汇率预测**: 外汇实时数据
- **波动率建模**: 期权数据
- **数据源**: Yahoo Finance, Wind

### 3. 工业物联网
- **设备预测维护**: 振动、温度、压力
- **质量检测**: 生产线传感器
- **能耗优化**: 工厂电力数据
- **数据源**: 工业数据库、MES系统

### 4. 医疗健康
- **心率预测**: 可穿戴设备数据
- **血糖预测**: 连续血糖监测
- **疾病预警**: 生命体征数据
- **数据源**: MIMIC, PhysioNet

### 5. 智能交通
- **交通流量**: 道路传感器
- **拥堵预测**: GPS轨迹
- **事故预警**: 交通摄像头
- **数据源**: 交通管理局、开源数据集

### 6. 能源系统
- **负荷预测**: 电力负荷数据
- **光伏预测**: 太阳能发电数据
- **储能优化**: 电池SOC预测
- **数据源**: 电网公司、太阳能电站

---

## 📊 数据获取指南

### 免费数据源
```python
# 气象数据
import requests

# Open-Meteo (免费无需API Key)
url = "https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&current_weather=true"

# 金融数据 (yfinance)
import yfinance as yf
data = yf.download("AAPL", start="2020-01-01")

# 交通数据 (NYC Taxi)
# https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
```

### 数据格式要求
```
格式: CSV, JSON, Parquet
要求:
- 时间戳列
- 数值特征列
- 目标变量列
- 缺失值处理
```

---

## 🎯 快速集成新场景

```python
from axiom_os import PhysicalAI
import pandas as pd

# 1. 加载数据
data = pd.read_csv("your_data.csv")

# 2. 初始化模型
ai = PhysicalAI(
    in_dim=data.shape[1] - 1,
    hidden_dim=64,
    out_dim=1
)

# 3. 训练
ai.fit(data.features, data.target)

# 4. 预测
predictions = ai.predict(new_data)
```

---

## 🤝 贡献新场景

欢迎贡献新的使用场景！请参考：
- [CONTRIBUTING.md](CONTRIBUTING.md)
- 创建 `axiom_os/experiments/your_scenario.py`
- 添加对应测试

---

## 📞 数据合作

如果您有真实数据并希望合作：
1. 联系项目维护者
2. 讨论数据脱敏方案
3. 共同开发新场景
