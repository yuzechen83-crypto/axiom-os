"""Test SPNN reinforcement calibration system"""
import torch
from spnn.boundary import BoundaryFluxOperator, DirichletBC, NeumannBC
from spnn.thermodynamics import EntropyConsistencyEnforcer
from spnn.axiom import AxiomCredibilityManager, ErrorAttributionDiagnostic
from spnn.training import SPNNCompleteLoss, LossWeights
from spnn.wrappers import ThermodynamicallySafeModule, DASWithDiagnostic


def test_boundary():
    op = BoundaryFluxOperator()
    u_ai = torch.randn(8, 4)
    u_prescribed = torch.zeros(8, 4)
    L = op.u_dirichlet(u_ai, u_prescribed, alpha=1.0)
    assert L.item() > 0
    print("Boundary OK")


def test_entropy():
    S = lambda x: -(x ** 2).sum()
    enf = EntropyConsistencyEnforcer(S, tolerance=1e-5)
    state = torch.randn(4, 3, requires_grad=True)
    dynamics = torch.randn(4, 3) * 0.1 - state * 0.1
    result = enf.check_entropy_production(dynamics, state)
    print("Entropy:", result.entropy_production, result.violation)
    corrected = enf.enforce_constraint(dynamics, state)
    print("Entropy OK")


def test_credibility():
    hard = lambda x, c=None: x * 0.5
    mgr = AxiomCredibilityManager(hard, calibration_window=50)
    for _ in range(10):
        inp = torch.randn(4, 3)
        out = inp * 0.5 + torch.randn(4, 3) * 0.1
        mgr.update_credibility({"input": inp, "output": out}, context={})
    print("Credibility beta:", mgr.beta)
    print("Credibility OK")


def test_error_attribution():
    diag = ErrorAttributionDiagnostic()
    soft = torch.randn(4, 8) * 2
    hard = torch.randn(4, 8) * 0.1
    gate = torch.tensor(0.9)
    report = diag.diagnose_violation(soft, hard, gate)
    print("Root cause:", report.root_cause)
    print("Error attribution OK")


def test_complete_loss():
    loss_fn = SPNNCompleteLoss(weights=LossWeights())
    pred = torch.randn(8, 2)
    target = torch.randn(8, 2)
    soft = torch.randn(8, 2)
    hard = torch.randn(8, 2) * 0.5
    losses = loss_fn(pred, target, soft_out=soft, hard_out=hard, step=100)
    assert "total" in losses
    print("Complete loss:", losses["total"].item())
    print("Complete loss OK")


if __name__ == "__main__":
    test_boundary()
    test_entropy()
    test_credibility()
    test_error_attribution()
    test_complete_loss()
    print("All reinforcement tests passed.")
