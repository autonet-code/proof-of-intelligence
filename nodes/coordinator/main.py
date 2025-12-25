"""
Autonet Coordinator Node

Verifies task completion by comparing solver solutions to ground truth.
Submits verification reports to the blockchain.

Implements:
- Bittensor-style multi-coordinator voting (Yuma Consensus)
- Gensyn-style checkpoint verification
- Truebit-style forced error detection
"""

import logging
import json
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..core import Node, NodeRole, DEFAULT_CONSTITUTION
from ..common import BlockchainInterface, IPFSClient, hash_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of verification."""
    task_id: int
    solver_address: str
    is_correct: bool
    score: int  # 0-100
    report_cid: str
    details: Dict[str, Any]
    is_forced_error: bool = False  # Did we detect a forced error?


@dataclass
class CheckpointVerification:
    """Result of checkpoint comparison."""
    step_number: int
    weights_match: bool
    data_indices_match: bool
    seed_match: bool
    first_divergence_step: Optional[int] = None


class CoordinatorNode(Node):
    """
    Coordinator node that verifies task solutions.

    Responsibilities:
    - Download ground truth and solver solutions from IPFS
    - Compare solutions against ground truth
    - Verify checkpoints for Gensyn-style dispute resolution
    - Detect forced errors for Truebit-style incentives
    - Submit verification votes to blockchain (multi-coordinator mode)
    - Participate in Yuma consensus
    """

    def __init__(
        self,
        blockchain: Optional[BlockchainInterface] = None,
        ipfs: Optional[IPFSClient] = None,
        forced_error_detector: bool = True,
        **kwargs,
    ):
        super().__init__(role=NodeRole.COORDINATOR, **kwargs)
        self.blockchain = blockchain or BlockchainInterface()
        self.ipfs = ipfs or IPFSClient()
        self.verification_history: List[VerificationResult] = []
        self.forced_error_detector = forced_error_detector
        self.ema_bond_strength: float = 0.5  # Start with neutral bond

    def verify_solution(
        self,
        task_id: int,
        solver_address: str,
        solution_cid: str,
        ground_truth_cid: str,
        known_forced_error_hash: Optional[str] = None,
    ) -> Optional[VerificationResult]:
        """
        Verify a solver's solution against the ground truth.

        Args:
            task_id: The task being verified
            solver_address: Address of the solver
            solution_cid: IPFS CID of the solver's solution
            ground_truth_cid: IPFS CID of the ground truth
            known_forced_error_hash: Hash of known forced error (if this is a trap task)

        Returns:
            Verification result if successful
        """
        logger.info(f"Verifying solution for task {task_id} from {solver_address[:10]}...")

        # Download solution
        solution = self.ipfs.get_json(solution_cid)
        if not solution:
            logger.error(f"Failed to download solution: {solution_cid}")
            return None

        # Download ground truth
        ground_truth = self.ipfs.get_json(ground_truth_cid)
        if not ground_truth:
            logger.error(f"Failed to download ground truth: {ground_truth_cid}")
            return None

        # Check for forced error first (Truebit-style)
        is_forced_error = False
        if self.forced_error_detector and known_forced_error_hash:
            solution_hash = hash_string(solution_cid)
            if solution_hash == known_forced_error_hash:
                logger.warning(f"FORCED ERROR DETECTED for task {task_id}!")
                is_forced_error = True
                # This solution is deliberately wrong - we caught the trap

        # Perform verification
        is_correct, score, details = self._compare_solutions(solution, ground_truth)

        # If we detected a forced error, mark as incorrect regardless of metrics
        if is_forced_error:
            is_correct = False
            details["forced_error_detected"] = True

        # Create verification report
        report = {
            "task_id": task_id,
            "solver": solver_address,
            "is_correct": is_correct,
            "score": score,
            "solution_cid": solution_cid,
            "ground_truth_cid": ground_truth_cid,
            "details": details,
            "is_forced_error": is_forced_error,
            "coordinator_bond_strength": self.ema_bond_strength,
        }

        # Upload report to IPFS
        report_cid = self.ipfs.add_json(report)
        if not report_cid:
            logger.error("Failed to upload verification report")
            return None

        result = VerificationResult(
            task_id=task_id,
            solver_address=solver_address,
            is_correct=is_correct,
            score=score,
            report_cid=report_cid,
            details=details,
            is_forced_error=is_forced_error,
        )

        self.verification_history.append(result)
        logger.info(f"Verification complete: correct={is_correct}, score={score}")

        return result

    def verify_checkpoints(
        self,
        task_id: int,
        solver_address: str,
        solver_checkpoints: List[Dict[str, Any]],
        reference_checkpoints: List[Dict[str, Any]],
    ) -> CheckpointVerification:
        """
        Verify checkpoints between solver and reference (Gensyn-style).
        This enables pinpointing the first divergent step in disputes.

        Args:
            task_id: The task being verified
            solver_address: Address of the solver
            solver_checkpoints: Checkpoints from solver
            reference_checkpoints: Reference checkpoints (from another verifier or re-computation)

        Returns:
            CheckpointVerification result with first divergence point
        """
        logger.info(f"Verifying {len(solver_checkpoints)} checkpoints for task {task_id}")

        first_divergence = None

        for i, (solver_cp, ref_cp) in enumerate(zip(solver_checkpoints, reference_checkpoints)):
            weights_match = solver_cp.get("weights_hash") == ref_cp.get("weights_hash")
            data_match = solver_cp.get("data_indices_hash") == ref_cp.get("data_indices_hash")
            seed_match = solver_cp.get("random_seed") == ref_cp.get("random_seed")

            if not (weights_match and data_match and seed_match):
                first_divergence = solver_cp.get("step_number", i)
                logger.warning(
                    f"Checkpoint divergence at step {first_divergence}: "
                    f"weights={weights_match}, data={data_match}, seed={seed_match}"
                )
                break

        return CheckpointVerification(
            step_number=solver_checkpoints[-1].get("step_number", len(solver_checkpoints)) if solver_checkpoints else 0,
            weights_match=first_divergence is None,
            data_indices_match=first_divergence is None,
            seed_match=first_divergence is None,
            first_divergence_step=first_divergence,
        )

    def _compare_solutions(
        self,
        solution: Dict[str, Any],
        ground_truth: Dict[str, Any],
    ) -> tuple:
        """
        Compare a solution against ground truth.

        Returns:
            (is_correct, score, details)
        """
        details = {}

        # Check if metrics exist
        sol_metrics = solution.get("metrics", {})
        gt_metrics = ground_truth.get("metrics", {})

        if not sol_metrics:
            return False, 0, {"error": "No metrics in solution"}

        # Compare accuracy
        sol_acc = sol_metrics.get("accuracy", 0)
        gt_acc = gt_metrics.get("accuracy", 0.9)

        accuracy_diff = abs(sol_acc - gt_acc)
        is_close = accuracy_diff < 0.1

        score = max(0, int(100 * (1 - accuracy_diff / gt_acc))) if gt_acc > 0 else 0

        details = {
            "solution_accuracy": sol_acc,
            "expected_accuracy": gt_acc,
            "accuracy_difference": accuracy_diff,
        }

        # Check for checkpoint consistency if available
        if "checkpoint_frequency" in solution:
            details["has_checkpoints"] = True
            details["checkpoint_frequency"] = solution["checkpoint_frequency"]

        # Check for RepOps version (determinism indicator)
        if "repops_version" in solution:
            details["repops_version"] = solution["repops_version"]
            details["deterministic"] = True

        is_correct = is_close and score >= 70

        return is_correct, score, details

    def submit_vote(self, result: VerificationResult) -> bool:
        """
        Submit verification vote to the blockchain (multi-coordinator mode).
        This participates in Yuma consensus.

        Args:
            result: The verification result to submit as a vote

        Returns:
            True if successful
        """
        logger.info(f"Submitting vote for task {result.task_id}")
        logger.info(f"  Vote: {'CORRECT' if result.is_correct else 'INCORRECT'}, Score: {result.score}")

        # In production, call ResultsRewards.submitVote
        # with (task_id, solver, is_correct, score, report_cid)

        return True

    def submit_verification(self, result: VerificationResult) -> bool:
        """
        Submit verification result to the blockchain (legacy single-coordinator mode).

        Args:
            result: The verification result to submit

        Returns:
            True if successful
        """
        logger.info(f"Submitting verification for task {result.task_id}")

        # In production, call ResultsRewards.submitVerification
        # with (task_id, solver, is_correct, score, report_cid)

        return True

    def report_forced_error(self, task_id: int, solution_hash: str) -> bool:
        """
        Report a detected forced error to claim jackpot.

        Args:
            task_id: The task with forced error
            solution_hash: Hash of the bad solution

        Returns:
            True if successful
        """
        logger.info(f"Reporting forced error for task {task_id}")
        logger.info(f"  Solution hash: {solution_hash[:20]}...")

        # In production, call ForcedErrorRegistry.reportForcedError

        return True

    def update_bond_strength(self, aligned_with_consensus: bool):
        """
        Update the coordinator's EMA bond strength based on consensus alignment.
        This affects future reward multipliers.
        """
        decay = 0.9
        current_value = 1.0 if aligned_with_consensus else 0.0
        self.ema_bond_strength = decay * self.ema_bond_strength + (1 - decay) * current_value

        logger.info(
            f"Bond strength updated: {self.ema_bond_strength:.4f} "
            f"(aligned={aligned_with_consensus})"
        )

    def get_bond_multiplier(self) -> float:
        """
        Get the reward multiplier based on current bond strength.
        Strong bonds (high consistency) get up to 1.5x rewards.
        """
        base_multiplier = 1.0
        max_bonus = 0.5  # Up to 50% bonus
        return base_multiplier + (self.ema_bond_strength * max_bonus)


def main():
    """Run the coordinator node."""
    node = CoordinatorNode()

    # Demo: verify a solution with multi-coordinator voting
    result = node.verify_solution(
        task_id=1,
        solver_address="0x1234567890abcdef...",
        solution_cid="QmSolution...",
        ground_truth_cid="QmGroundTruth...",
    )

    if result:
        # Submit as vote in multi-coordinator mode
        node.submit_vote(result)

        # Show bond multiplier
        multiplier = node.get_bond_multiplier()
        print(f"\nVerification result: {result}")
        print(f"Bond strength: {node.ema_bond_strength:.4f}")
        print(f"Reward multiplier: {multiplier:.2f}x")

    # Demo: checkpoint verification
    solver_checkpoints = [
        {"step_number": 10, "weights_hash": "abc123", "data_indices_hash": "def456", "random_seed": "seed1"},
        {"step_number": 20, "weights_hash": "abc124", "data_indices_hash": "def457", "random_seed": "seed2"},
    ]
    reference_checkpoints = [
        {"step_number": 10, "weights_hash": "abc123", "data_indices_hash": "def456", "random_seed": "seed1"},
        {"step_number": 20, "weights_hash": "abc124", "data_indices_hash": "def457", "random_seed": "seed2"},
    ]

    checkpoint_result = node.verify_checkpoints(
        task_id=1,
        solver_address="0x1234567890abcdef...",
        solver_checkpoints=solver_checkpoints,
        reference_checkpoints=reference_checkpoints,
    )

    print(f"\nCheckpoint verification: {checkpoint_result}")

    # Run for a few cycles
    node.run(max_cycles=3)


if __name__ == "__main__":
    main()
