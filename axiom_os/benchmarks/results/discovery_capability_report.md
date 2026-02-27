# Discovery 发现能力基准报告

**时间**: 2026-02-25 18:45:04

## 评估目标

在混沌复杂的现实数据（SPARC）下，评估发现的公式与**已知物理定律**的相似度。
高相似度证明系统具有从复杂现实中复现规律的能力。

## 已知定律 (Ground Truth)

- **rar**: g_obs = g_bar * nu(g_bar/g†), nu 为 g_bar 的普适函数
- **theory_validator**: V_def^2 = C * V_bary_sq / r^alpha (rho 形式)
- **inverse_projection**: Sigma_halo = C * Sigma_baryon / r^beta

## 相似度结果

### rar
- **相似度**: 100.0%
- 发现公式: `-7.611*log_g_bar + -4.495e-05*g_bar + 1.071*log_g_bar^2 + 14.53`
- 已知定律: g_obs = g_bar * nu(g_bar/g†), nu 为 g_bar 的普适函数

  细节: {'has_g_bar': True, 'has_log_g_bar': True, 'g_dagger': 645.6198100058118, 'g_dagger_valid': True}

### theory_validator
- **相似度**: 50.4%
- 发现公式: `474.1*r + -0.148*V_bary_sq + -1.541*r^2 + 1.146e-06*V_bary_sq^2 + 0.01754*r*V_ba...`
- 已知定律: V_def^2 = C * V_bary_sq / r^alpha (rho 形式)

  细节: {'has_v_bary_sq': True, 'has_r': True, 'has_rho_form': False}

### inverse_projection
- **相似度**: 38.2%
- 发现公式: `0.06808*Sigma_baryon + -44.55*r + 0.644*r^2 + -0.05165*Sigma_baryon*r + 580.6`
- 已知定律: Sigma_halo = C * Sigma_baryon / r^beta

  细节: {'has_sigma_baryon': True, 'has_r': True, 'has_ratio_form': False}


## 综合

**平均相似度**: 62.9%

---
*Axiom-OS Discovery 发现能力基准*