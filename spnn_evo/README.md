# SPNN-Evo 2.0 (Axiom-OS Ecosystem)

**Digital Scientific Community** - AI for Science

| Metaphor | Component | Role |
|----------|-----------|------|
| **Hunter** | Neural Network | Explores the unknown (residuals) |
| **Library** | Hippocampus | Stores crystallized physical laws |
| **Language** | UPI | Cross-disciplinary protocol |
| **Reactor** | RCLN | Interdisciplinary coupling |

## Dual-Loop Architecture

- **Loop A (Construction)**: Fast inference = F_hard(Known) + F_soft(Neural Correction)
- **Loop B (Discovery)**: Extract F_soft → Buckingham π → Symbolic Law → Crystallize → Reset

## File Structure

```
spnn_evo/
├── core/
│   ├── upi.py         # UPIState: tensor, units [M,L,T,Q,Θ], spacetime, semantics
│   └── hippocampus.py # Library: Semantic_ID → Symbolic_Expression
├── layers/
│   └── rcln.py        # RCLN: F_hard + F_soft, Activity Monitor → Discovery Hotspot
├── engine/
│   └── discovery.py   # Buckingham π, Symbolic Regression, Crystallization
└── main.py            # Ecosystem: Domain_A ↔ RCLN ↔ Domain_B
```

## Run

```bash
.\.venv\Scripts\python.exe -m spnn_evo.main
# or
.\.venv\Scripts\python.exe run_spnn_evo.py
```

## UPIState Usage

```python
from spnn_evo.core.upi import UPIState, VELOCITY

u = UPIState(tensor, VELOCITY, spacetime=(t,x,y,z), semantics="PlasmaDensity")
v = u + u  # Units must match
u.assert_causality(other)  # Light cone check
```
