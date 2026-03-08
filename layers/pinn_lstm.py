"""
Physics-Informed LSTM (PINN-LSTM)
Temporal modeling with optional physics residual loss.
Integrates with Axiom Hard Core + Coach.
"""

from typing import Optional, Callable, Tuple
import torch
import torch.nn as nn


class LSTMSoftShell(nn.Module):
    """
    LSTM-based Soft Shell for sequential/temporal data.
    Input: (B, seq_len, input_dim)
    Output: (B, output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        lstm_out = hidden_dim * 2 if bidirectional else hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(lstm_out, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, input_dim)
        returns: (B, output_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)
        out, (h_n, _) = self.lstm(x)
        # Use last hidden state: h_n (num_layers, B, hidden_dim)
        last_h = h_n[-1]  # (B, hidden_dim)
        if self.lstm.bidirectional:
            # Concatenate forward and backward
            last_h_fw = h_n[-2]
            last_h_bw = h_n[-1]
            last_h = torch.cat([last_h_fw, last_h_bw], dim=-1)
        return self.proj(last_h)


class PhysicsInformedLSTM(nn.Module):
    """
    Physics-Informed LSTM: Hard Core + LSTM residual.
    y_total = y_hard + lambda_res * y_soft

    For turbulence: input (B, seq_len, 4) = (t, x, y, z) sequences.
    Hard core predicts (u, v) from last (t,x,y,z); LSTM adds temporal residual.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_len: int,
        hard_core_func: Optional[Callable] = None,
        lambda_res: float = 1.0,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.hard_core = hard_core_func
        self.lambda_res = lambda_res
        self._lambda_res_init = lambda_res

        self.lstm_shell = LSTMSoftShell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        )

    def set_lambda_res(self, value: float) -> None:
        self.lambda_res = float(value)

    def set_lambda_decay(self, epoch: int, total_epochs: int, decay_min: float = 0.5) -> None:
        if total_epochs <= 0:
            return
        frac = min(1.0, epoch / total_epochs)
        self.lambda_res = self._lambda_res_init * (1.0 - (1.0 - decay_min) * frac)

    def forward(
        self,
        x_seq: torch.Tensor,
        x_last: Optional[torch.Tensor] = None,
        target_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x_seq: (B, seq_len, input_dim) - full sequence
        x_last: (B, input_dim) - last step (for hard core). If None, use x_seq[:, -1].
        target_coords: (B, 4) - optional target (t,x,y,z) for hard core when forecasting.
        Returns: (B, output_dim)
        """
        if target_coords is not None:
            x_last = target_coords
        elif x_last is None:
            x_last = x_seq[:, -1, :]

        # Hard core: physics from last point
        if self.hard_core is not None:
            y_hard = self.hard_core(x_last)
            if not isinstance(y_hard, torch.Tensor):
                y_hard = torch.as_tensor(y_hard, dtype=torch.float32, device=x_seq.device)
            if y_hard.dim() == 1:
                y_hard = y_hard.unsqueeze(0)
        else:
            y_hard = torch.zeros(x_seq.shape[0], self.output_dim, device=x_seq.device, dtype=x_seq.dtype)

        # LSTM residual
        y_soft = self.lstm_shell(x_seq)
        return y_hard + self.lambda_res * y_soft


def pde_residual_loss_temporal(
    u_pred: torch.Tensor,
    u_prev: torch.Tensor,
    dt: float = 1e-2,
) -> torch.Tensor:
    """
    Simplified temporal continuity: u_t ≈ (u - u_prev) / dt.
    Encourages smooth temporal evolution.
    u_pred, u_prev: (B, 2) for (u, v)
    """
    if dt <= 0:
        return torch.tensor(0.0, device=u_pred.device)
    residual = (u_pred - u_prev) / dt
    return (residual ** 2).mean()
