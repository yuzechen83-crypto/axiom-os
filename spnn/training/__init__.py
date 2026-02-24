"""SPNN Training: Multi-objective loss, complete loss, intelligent protocol, turbulence"""

from .losses import SPNNLoss
from .complete_loss import SPNNCompleteLoss, LossWeights
from .trainer import SPNNTrainer
from .turbulence import (
    run_turbulence_training,
    TurbulenceConfig,
    TurbulenceLoss,
    generate_burgers_turbulence,
    generate_2d_turbulence,
)
