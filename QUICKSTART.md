# Axiom-OS x NVIDIA Isaac Sim - 快速开始指南

## 环境配置状态

✅ **已完成配置：**
- Isaac Sim 4.0+ 安装在 `C:\Users\ASUS\isaacsim`
- numpy 已安装在 Isaac Sim Python 中
- torch 已安装在 Isaac Sim Python 中
- 所有脚本已创建并配置

## 运行方式

### 方法 1: 使用 Kit 启动器 (推荐)

```batch
# 运行测试模式 (100步)
.\run_with_kit.bat --test

# 运行完整模拟 (2000步)
.\run_with_kit.bat

# 无渲染模式 (更快)
.\run_with_kit.bat --headless
```

### 方法 2: 直接使用 Kit

```batch
cd C:\Users\ASUS\isaacsim\_build\windows-x86_64\release

# 运行演示
.\kit\kit.exe apps\isaacsim.exp.full.kit --exec python C:\Users\ASUS\PycharmProjects\PythonProject1\run_isaac_demo.py --test
```

### 方法 3: Headless 模式 (无 GUI，最快)

```batch
# 添加 headless 参数
.\kit\kit.exe apps\isaacsim.exp.full.kit --/app/window/width=1 --/app/window/height=1 --exec python C:\Users\ASUS\PycharmProjects\PythonProject1\run_isaac_demo.py --test
```

## 首次运行注意事项

1. **加载时间**: Isaac Sim 首次启动需要 1-3 分钟加载扩展
2. **GPU 检测**: 已检测到 NVIDIA RTX 5070 Laptop GPU
3. **许可证**: 确保已接受 EULA (`C:\Users\ASUS\isaacsim\.eula_accepted`)

## 测试 Isaac Sim 环境

```batch
# 测试环境是否可用
.\run_with_kit.bat test_isaac_simple.py
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `simulation/isaac_env.py` | Isaac Sim 环境包装器 |
| `run_isaac_demo.py` | 主演示脚本 |
| `run_with_kit.bat` | Kit 启动器 |
| `test_isaac_simple.py` | 环境测试脚本 |

## 故障排除

### "No module named 'omni.isaac'"
- 必须使用 Kit 启动器运行，不能直接运行 Python
- Kit 会自动加载所有必需的扩展

### 启动时间过长
- 首次启动需要编译 shader 缓存
- 使用 `--headless` 模式可跳过 GUI

### 内存不足
- Isaac Sim 需要至少 8GB 显存
- RTX 5070 Laptop (8GB) 可以运行
- 关闭其他应用程序释放内存

## 下一步

运行成功后，可以：
1. 调整 `Go1Config` 参数测试不同场景
2. 加载预训练的 Axiom 模型
3. 启用视频录制保存演示
4. 修改混沌注入器参数测试鲁棒性
