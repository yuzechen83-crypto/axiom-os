"""
Axiom-OS 真实数据场景扩展

包含：气象、金融、工业、医疗、交通、能源等场景
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import warnings

# ============== 气象数据 ==============

def load_weather_data(
    latitude: float = 52.52,
    longitude: float = 13.41,
    days: int = 30,
    use_synthetic_if_fail: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    加载气象数据（Open-Meteo API）
    
    Args:
        latitude: 纬度
        longitude: 经度
        days: 数据天数
        use_synthetic_if_fail: API失败时使用合成数据
    
    Returns:
        (data, metadata) - 数据和元信息
    """
    try:
        import requests
        
        url = f"https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "forecast_days": days
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        temperature = np.array(data["hourly"]["temperature_2m"])
        humidity = np.array(data["hourly"]["relative_humidity_2m"])
        wind_speed = np.array(data["hourly"]["wind_speed_10m"])
        
        # 合并特征
        X = np.column_stack([temperature, humidity, wind_speed])
        
        # 移除NaN
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        
        metadata = {
            "source": "Open-Meteo",
            "location": f"{latitude}, {longitude}",
            "n_samples": len(X),
            "features": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
        }
        
        return X, metadata
        
    except Exception as e:
        if use_synthetic_if_fail:
            warnings.warn(f"Using synthetic weather data: {e}")
            return generate_synthetic_weather(days)
        raise


def generate_synthetic_weather(days: int = 30) -> Tuple[np.ndarray, Dict]:
    """生成合成气象数据"""
    n_samples = days * 24  # 每小时一个样本
    
    # 温度：日周期 + 噪声
    t = np.linspace(0, days * 2 * np.pi, n_samples)
    temperature = 20 + 10 * np.sin(t) + np.random.randn(n_samples) * 2
    
    # 湿度：与温度负相关
    humidity = 60 - 0.5 * temperature + np.random.randn(n_samples) * 10
    humidity = np.clip(humidity, 0, 100)
    
    # 风速：随机
    wind_speed = np.abs(np.random.randn(n_samples) * 5 + 10)
    
    X = np.column_stack([temperature, humidity, wind_speed])
    
    metadata = {
        "source": "synthetic",
        "n_samples": n_samples,
        "features": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
    }
    
    return X, metadata


# ============== 金融数据 ==============

def load_stock_data(
    symbol: str = "AAPL",
    start_date: str = "2020-01-01",
    end_date: str = "2024-01-01",
    use_synthetic_if_fail: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    加载股票数据 (yfinance)
    
    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        use_synthetic_if_fail: 失败时使用合成数据
    
    Returns:
        (data, metadata) - 数据和元信息
    """
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        # 使用特征：开盘价、收盘价、最高价、最低价、成交量变化
        features = df[['Open', 'Close', 'High', 'Low', 'Volume']].values
        
        # 计算收益率作为目标
        returns = df['Close'].pct_change().values
        
        # 特征：滞后收益 + 技术指标
        X = np.column_stack([
            features,
            returns
        ])
        
        # 移除NaN
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        
        metadata = {
            "source": "yfinance",
            "symbol": symbol,
            "n_samples": len(X),
            "features": ["Open", "Close", "High", "Low", "Volume", "returns"]
        }
        
        return X, metadata
        
    except Exception as e:
        if use_synthetic_if_fail:
            warnings.warn(f"Using synthetic stock data: {e}")
            return generate_synthetic_stock(1000)
        raise


def generate_synthetic_stock(n_samples: int = 1000) -> Tuple[np.ndarray, Dict]:
    """生成合成股票数据"""
    # 随机游走
    returns = np.random.randn(n_samples) * 0.02
    price = 100 * np.exp(np.cumsum(returns))
    
    # 生成OHLCV
    high = price * (1 + np.abs(np.random.randn(n_samples) * 0.01))
    low = price * (1 - np.abs(np.random.randn(n_samples) * 0.01))
    open_price = price * (1 + np.random.randn(n_samples) * 0.005)
    close_price = price
    volume = np.random.randint(1000000, 10000000, n_samples)
    
    X = np.column_stack([open_price, close_price, high, low, volume, returns])
    
    metadata = {
        "source": "synthetic",
        "n_samples": n_samples,
        "features": ["Open", "Close", "High", "Low", "Volume", "returns"]
    }
    
    return X, metadata


# ============== 工业IoT数据 ==============

def load_industrial_sensor(
    n_samples: int = 10000,
    noise_level: float = 0.1
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    生成工业传感器数据（模拟设备预测维护）
    
    特征：振动、温度、压力、转速、功率
    目标：预测设备健康状态
    """
    # 时间序列
    t = np.linspace(0, 100, n_samples)
    
    # 正常模式 + 退化模式
    degradation = np.linspace(0, 0.5, n_samples)
    
    # 振动
    vibration = 0.1 + 0.05 * np.sin(t) + degradation * 0.3 + np.random.randn(n_samples) * noise_level
    
    # 温度
    temperature = 50 + 10 * np.sin(t * 0.1) + degradation * 20 + np.random.randn(n_samples) * noise_level * 5
    
    # 压力
    pressure = 100 + 5 * np.sin(t * 0.2) + degradation * 10 + np.random.randn(n_samples) * noise_level * 3
    
    # 转速
    rpm = 3000 + 100 * np.sin(t * 0.05) - degradation * 500 + np.random.randn(n_samples) * noise_level * 50
    
    # 功率
    power = 50 + 5 * np.sin(t * 0.15) + degradation * 15 + np.random.randn(n_samples) * noise_level * 2
    
    # 目标：健康指数 (1=健康, 0=故障)
    health = 1 - degradation + np.random.randn(n_samples) * 0.05
    health = np.clip(health, 0, 1)
    
    X = np.column_stack([vibration, temperature, pressure, rpm, power, health])
    
    metadata = {
        "source": "synthetic_industrial",
        "n_samples": n_samples,
        "features": ["vibration", "temperature", "pressure", "rpm", "power", "health_index"]
    }
    
    return X, metadata


# ============== 能源数据 ==============

def load_energy_consumption(
    n_days: int = 365,
    hourly: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    生成能源消耗数据
    
    特征：时间特征、天气特征、历史负荷
    目标：预测未来负荷
    """
    if hourly:
        n_samples = n_days * 24
    else:
        n_samples = n_days
    
    # 时间
    t = np.linspace(0, n_days * 2 * np.pi, n_samples)
    
    # 日周期 + 周周期 + 年周期
    daily = np.sin(t)
    weekly = np.sin(t / 7 * 2 * np.pi)
    yearly = np.sin(t / 365 * 2 * np.pi)
    
    # 温度效应
    temperature = 15 + 10 * yearly + np.random.randn(n_samples) * 2
    
    # 基础负荷 + 周期效应 + 温度效应 + 随机
    load = (
        100 + 
        30 * daily + 
        20 * weekly +
        2 * temperature +
        np.random.randn(n_samples) * 5
    )
    
    # 目标：下一时刻负荷
    load_next = np.roll(load, -1)
    load_next[-1] = load[-1]  # 最后一个值
    
    X = np.column_stack([
        load,
        daily,
        weekly,
        yearly,
        temperature,
        load_next  # 目标
    ])
    
    metadata = {
        "source": "synthetic_energy",
        "n_samples": n_samples,
        "features": ["load", "daily_pattern", "weekly_pattern", "yearly_pattern", "temperature", "target_load"]
    }
    
    return X, metadata


# ============== 主函数 ==============

def load_dataset(
    dataset_name: str,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    统一数据加载接口
    
    Args:
        dataset_name: 数据集名称
            - "weather": 气象数据
            - "stock": 股票数据
            - "industrial": 工业传感器
            - "energy": 能源消耗
        **kwargs: 其他参数
    
    Returns:
        (X, metadata)
    """
    datasets = {
        "weather": load_weather_data,
        "stock": load_stock_data,
        "industrial": load_industrial_sensor,
        "energy": load_energy_consumption,
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(datasets.keys())}")
    
    return datasets[dataset_name](**kwargs)


if __name__ == "__main__":
    # 测试各数据集
    print("Testing weather data...")
    X, meta = load_dataset("weather", days=7)
    print(f"  Shape: {X.shape}, Source: {meta['source']}")
    
    print("Testing industrial sensor...")
    X, meta = load_dataset("industrial", n_samples=1000)
    print(f"  Shape: {X.shape}, Source: {meta['source']}")
    
    print("Testing energy consumption...")
    X, meta = load_dataset("energy", n_days=30)
    print(f"  Shape: {X.shape}, Source: {meta['source']}")
