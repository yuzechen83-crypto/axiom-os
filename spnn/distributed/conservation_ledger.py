"""
Conservation Ledger 弹性守恒协议
全局守恒账本、定期收支平衡、投票协商、BFT 最终性
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class VoteAction(Enum):
    ACCEPT = "accept"
    CORRECT = "correct"
    INVESTIGATE = "investigate"


@dataclass
class ConservationBundle:
    """守恒量束: 当前容忍度随时间变化"""
    level: str
    base_tolerance: float

    def current_tolerance(self, t: float) -> float:
        return self.base_tolerance


class ConservationLedger:
    """
    全局守恒账本
    record_transaction, reconcile, collect_votes, BFT FinalDecision
    """

    def __init__(
        self,
        conservation_bundles: Optional[Dict[str, ConservationBundle]] = None,
        nodes: Optional[List[Any]] = None,
        bft_threshold: float = 2.0 / 3.0,
    ):
        self.ledger: Dict[str, Dict] = {}
        self.bundles = conservation_bundles or {}
        self.nodes = nodes or []
        self.vote_history: List[Dict] = []
        self.bft_threshold = bft_threshold

    def record_transaction(
        self,
        node_id: str,
        quantity: str,
        change: float,
        timestamp: float,
    ) -> None:
        if quantity not in self.ledger:
            self.ledger[quantity] = {"credit": [], "debit": [], "balance": 0.0}
        if change > 0:
            self.ledger[quantity]["credit"].append({"node": node_id, "amount": change, "time": timestamp})
        else:
            self.ledger[quantity]["debit"].append({"node": node_id, "amount": -change, "time": timestamp})
        self.ledger[quantity]["balance"] += change

    def reconcile(
        self,
        current_time: float,
        reconciliation_interval: float = 100.0,
    ) -> List[Dict]:
        """定期对账"""
        decisions = []
        for quantity, bundle in self.bundles.items():
            if quantity not in self.ledger:
                continue
            balance = self.ledger[quantity]["balance"]
            tolerance = bundle.current_tolerance(current_time)
            if abs(balance) > tolerance:
                votes = self.collect_votes(quantity, balance)
                decision = self.make_decision(votes, bundle.level)
                decisions.append({
                    "time": current_time,
                    "quantity": quantity,
                    "imbalance": balance,
                    "tolerance": tolerance,
                    "votes": votes,
                    "decision": decision,
                })
                self.vote_history.append(decisions[-1])
                if decision["action"] == "correct":
                    correction = self.compute_correction(balance, decision.get("strategy", "proportional"))
                    self.apply_correction(quantity, correction)
        return decisions

    def collect_votes(self, quantity: str, imbalance: float) -> Dict[str, int]:
        """收集各节点意见"""
        votes = {"accept": 0, "correct": 0, "investigate": 0}
        for node in self.nodes:
            if hasattr(node, "vote_on_imbalance"):
                v = node.vote_on_imbalance(quantity, imbalance)
                votes[v.value if hasattr(v, "value") else str(v)] = votes.get(v, 0) + 1
            else:
                votes["correct"] += 1
        return votes

    def make_decision(self, votes: Dict[str, int], level: str) -> Dict[str, Any]:
        """BFT: N_correct/N_total > 2/3 → Accept"""
        total = sum(votes.values()) or 1
        n_correct = votes.get("correct", 0) + votes.get("accept", 0)
        final_accept = n_correct / total > self.bft_threshold
        return {
            "action": "accept" if final_accept else "correct",
            "final_decision": "Accept" if final_accept else "Reject",
            "strategy": "proportional",
        }

    def compute_correction(self, balance: float, strategy: str) -> float:
        return -balance

    def apply_correction(self, quantity: str, correction: float) -> None:
        self.ledger[quantity]["balance"] += correction
