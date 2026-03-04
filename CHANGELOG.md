# Changelog

## [Unreleased]

### Changed

- **DeepSeek Agent 代码审查与清理 (14.3)**
  - 移除 chat_ui 中冗余的 `import os`
  - 修正 `_format_error_message` 的 `error_type` 类型注解为 `Optional[str]`
  - 修正 `_check_api_key` 参数类型为 `Optional[str]` 以支持 None

- **LLM 集成迁移：Gemini → DeepSeek**
  - 将 LLM 后端从 Google Gemini API 迁移到 DeepSeek API
  - 使用 OpenAI 兼容接口（`openai` Python SDK）
  - API 端点：`https://api.deepseek.com`
  - 默认模型：`deepseek-chat`
  - **破坏性变更**：环境变量从 `GEMINI_API_KEY` 更改为 `DEEPSEEK_API_KEY`

### Migration Guide

如果您之前使用 Gemini 集成，请按以下步骤迁移到 DeepSeek：

1. **获取 DeepSeek API 密钥**
   - 访问 [DeepSeek Platform](https://platform.deepseek.com)
   - 注册账号并生成 API 密钥

2. **更新环境变量**
   ```bash
   # 移除旧的 Gemini 环境变量
   unset GEMINI_API_KEY
   unset GOOGLE_API_KEY
   
   # 设置新的 DeepSeek 环境变量
   export DEEPSEEK_API_KEY=your_deepseek_api_key
   ```

3. **更新依赖**
   ```bash
   pip install --upgrade axiom-os
   # 或者如果从源码安装
   pip install -e ".[dev]"
   ```

4. **验证迁移**
   ```bash
   # 运行 Chat UI 测试
   streamlit run axiom_os/agent/chat_ui.py
   
   # 在界面中启用 "使用 DeepSeek 扩展" 并测试对话
   ```

5. **功能对比**
   - ✅ 所有工具调用功能保持不变
   - ✅ 对话历史和上下文管理保持不变
   - ✅ Chat UI 界面和交互流程保持不变
   - ⚠️ 环境变量名称已更改（见上文）
   - ⚠️ API 端点和模型名称已更改

6. **故障排除**
   - 如果遇到认证错误，请检查 `DEEPSEEK_API_KEY` 是否正确设置
   - 如果遇到连接错误，请确保可以访问 `https://api.deepseek.com`
   - 查看详细错误信息以获取具体的解决建议

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
