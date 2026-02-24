# Axiom-OS 性能基准测试报告

**配置**: standard
**时间**: 2026-02-24 12:50:13

## 结果汇总

| 指标 | 值 | 单位 | 备注 |
|------|-----|------|------|
| rcln_forward [latency_ms] | 0.2107 | ms | batch_size=1024 |
| rcln_forward [throughput] | 4859644.5370 | samples/s | batch_size=1024 |
| discovery_multivariate [latency_s] | 0.0005 | s | n_samples=200 |
| hippocampus_retrieve [latency_ms] | 0.0066 | ms | n_queries=100 |
| coach_score [latency_ms] | 0.2781 | ms | n_samples=10000 |
| coach_loss_torch [latency_ms] | 0.1858 | ms | n_samples=10000 |
| bundle_field_select [latency_ms] | 0.0113 | ms | n_queries=500 |
| turbulence_training [elapsed_s] | 0.3321 | s | epochs=200 |
| turbulence_training [epoch_time_ms] | 1.6605 | ms |  |
| rar_discovery [elapsed_s] | 0.6386 | s | n_galaxies=20, epochs=100, ok=True |
| rar_discovery [r2] | 0.6519 |  | n_samples=543 |
| e2e_quick_main [elapsed_s] | 24.1219 | s | ok=True |
| e2e_quick_rar [elapsed_s] | 1.1301 | s | ok=True |
| e2e_quick_battery [elapsed_s] | 7.9126 | s | ok=True |
| e2e_quick_total [elapsed_s] | 33.1645 | s |  |
| memory_peak [mb] | 0.1759 | MB |  |

## ✅ 无阈值告警
