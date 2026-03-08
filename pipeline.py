"""
Axiom-OS Unified Pipeline
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple

from axiom_os.engine.discovery import DiscoveryEngine
from axiom_os.core.hippocampus import Hippocampus
from axiom_os.layers.rcln import RCLNLayer


class AxiomPipeline:
    def __init__(self, domain: str = "mechanics", device: str = "auto"):
        self.domain = domain
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if device == "auto" else device
        self._init_components()
        self.is_fitted_ = False
        self.formula_ = None
        
    def _init_components(self):
        self.discovery = DiscoveryEngine(use_pysr=False)
        self.hippocampus = Hippocampus(dim=32, capacity=5000, use_semantic_rag=False)
        self.rcln = None
        
    def discover_formula(self, X: np.ndarray, y: np.ndarray) -> Tuple[str, Dict]:
        print("[Discovery] Discovering from", len(X), "samples...")
        formula = self.discovery.discover(X, y)
        
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            pred = X @ coeffs
            mse = np.mean((pred - y) ** 2)
            r2 = 1 - mse / (y.var() + 1e-8)
        except:
            r2, mse = 0, float('inf')
            
        metrics = {"r2": r2, "mse": mse}
        print("[Discovery] Found:", formula)
        return formula, metrics
    
    def fit(self, X: np.ndarray, y: np.ndarray, crystallize: bool = True) -> "AxiomPipeline":
        print("=" * 50)
        print("AXIOM-OS PIPELINE")
        print("=" * 50)
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        # Step 1: Discover
        print("\n[Step 1/2] Discovering formula...")
        formula, metrics = self.discover_formula(X, y)
        self.last_r2_ = metrics["r2"]
        
        # Step 2: Train RCLN (pure neural for now)
        print("\n[Step 2/2] Training RCLN...")
        
        self.rcln = RCLNLayer(
            input_dim=X.shape[1],
            hidden_dim=32,
            output_dim=1,
        ).to(self.device)
        
        self._train_rcln(X_tensor, y_tensor)
        
        # Store in hippocampus (knowledge)
        if crystallize and formula:
            formula_id = self.domain + "_" + str(len(self.hippocampus.knowledge_base))
            embedding = np.random.randn(32)
            info = {"formula": formula, "domain": self.domain, "r2": metrics.get("r2", 0)}
            self.hippocampus.store(embedding, info, formula_id)
            print("[Crystallize] Stored:", formula_id)
        
        self.is_fitted_ = True
        self.formula_ = formula
        print("\nDONE - Pipeline integrated!")
        return self
    
    def _train_rcln(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 100, lr: float = 0.01):
        optimizer = torch.optim.AdamW(self.rcln.parameters(), lr=lr)
        self.rcln.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self.rcln(X).squeeze()
            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 25 == 0:
                self.rcln.eval()
                with torch.no_grad():
                    mse = nn.functional.mse_loss(self.rcln(X).squeeze(), y).item()
                    r2 = 1 - mse / (y.var().item() + 1e-8)
                print("  Epoch", epoch+1, ": MSE=", mse, ", R2=", r2)
                self.rcln.train()
                
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Not fitted")
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.rcln.eval()
        with torch.no_grad():
            pred = self.rcln(X_tensor).squeeze().cpu().numpy()
        return pred
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        pred = self.predict(X)
        mse = np.mean((pred - y) ** 2)
        return {"mse": mse, "rmse": np.sqrt(mse), "r2": 1 - mse / (y.var() + 1e-8)}
    
    def get_knowledge(self) -> Dict:
        return {"formulas": list(self.hippocampus.knowledge_base.keys()), "count": len(self.hippocampus.knowledge_base)}
