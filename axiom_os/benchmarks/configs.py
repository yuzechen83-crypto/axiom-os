"""
Phase 4.3: 多配置 quick | standard | full
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class BenchConfig:
    """基准测试配置"""
    name: str
    unit: bool
    integration: bool
    e2e: bool
    memory: bool
    e2e_quick: bool
    # 集成参数
    turbulence_epochs: int
    rar_galaxies: int
    rar_epochs: int
    # E2E 参数
    main_t_max: int
    main_discover_interval: int


QUICK = BenchConfig(
    name="quick",
    unit=True,
    integration=True,
    e2e=False,
    memory=False,
    e2e_quick=False,
    turbulence_epochs=100,
    rar_galaxies=15,
    rar_epochs=50,
    main_t_max=100,
    main_discover_interval=30,
)

STANDARD = BenchConfig(
    name="standard",
    unit=True,
    integration=True,
    e2e=False,
    memory=True,
    e2e_quick=True,
    turbulence_epochs=200,
    rar_galaxies=20,
    rar_epochs=100,
    main_t_max=200,
    main_discover_interval=50,
)

FULL = BenchConfig(
    name="full",
    unit=True,
    integration=True,
    e2e=True,
    memory=True,
    e2e_quick=False,
    turbulence_epochs=500,
    rar_galaxies=30,
    rar_epochs=200,
    main_t_max=500,
    main_discover_interval=50,
)

CONFIG_MAP = {"quick": QUICK, "standard": STANDARD, "full": FULL}
