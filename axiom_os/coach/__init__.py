"""
Coach - SPNN-Evo 严肃教练
ChatGPT 式：Coach 对预测打分，辅助损失 L_coach = 1 - score
"""

from .spnn_evo_coach import coach_score, coach_score_batch, coach_loss_torch

__all__ = ["coach_score", "coach_score_batch", "coach_loss_torch"]
