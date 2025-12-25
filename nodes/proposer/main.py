"""
Autonet Proposer Node

Generates training tasks for distributed AI model improvement.
Implements the "Absolute Zero" task generation loop.
"""

import logging
import time
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..core import Node, NodeRole, DEFAULT_CONSTITUTION
from ..common import BlockchainInterface, IPFSClient, hash_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TaskSpec:
    """Specification for a training task."""
    project_id: int
    description: str
    input_data_cid: str
    expected_output_format: Dict[str, Any]
    difficulty: float
    ground_truth_cid: str
    learnability_reward: int
    solver_reward: int


class ProposerNode(Node):
    """
    Proposer node that generates training tasks.

    Responsibilities:
    - Generate training tasks with ground-truth solutions
    - Estimate task learnability
    - Submit tasks to the blockchain
    - Reveal ground truth after solvers submit
    """

    def __init__(
        self,
        blockchain: Optional[BlockchainInterface] = None,
        ipfs: Optional[IPFSClient] = None,
        **kwargs,
    ):
        super().__init__(role=NodeRole.PROPOSER, **kwargs)
        self.blockchain = blockchain or BlockchainInterface()
        self.ipfs = ipfs or IPFSClient()
        self.pending_tasks: Dict[int, TaskSpec] = {}

    def generate_task(
        self,
        project_id: int,
        description: str,
        input_data: Dict[str, Any],
        ground_truth: Dict[str, Any],
        learnability_reward: int = 10 * 10**18,
        solver_reward: int = 5 * 10**18,
    ) -> Optional[int]:
        """
        Generate and submit a new training task.

        Args:
            project_id: The project this task belongs to
            description: Human-readable task description
            input_data: Input data for the task
            ground_truth: Expected output (kept secret until reveal)
            learnability_reward: Reward for proposer (r_propose)
            solver_reward: Reward for solver (r_solve)

        Returns:
            Task ID if successful, None otherwise
        """
        logger.info(f"Generating task for project {project_id}")

        # Upload input data to IPFS
        input_cid = self.ipfs.add_json(input_data)
        if not input_cid:
            logger.error("Failed to upload input data to IPFS")
            return None

        # Upload ground truth to IPFS (for later reveal)
        ground_truth_cid = self.ipfs.add_json(ground_truth)
        if not ground_truth_cid:
            logger.error("Failed to upload ground truth to IPFS")
            return None

        # Create task spec
        task_spec = TaskSpec(
            project_id=project_id,
            description=description,
            input_data_cid=input_cid,
            expected_output_format={"type": "json"},
            difficulty=self._estimate_difficulty(input_data),
            ground_truth_cid=ground_truth_cid,
            learnability_reward=learnability_reward,
            solver_reward=solver_reward,
        )

        # Compute hashes for blockchain
        spec_hash = hash_string(json.dumps({
            "project_id": project_id,
            "description": description,
            "input_cid": input_cid,
        }))
        ground_truth_hash = hash_string(ground_truth_cid)

        # Submit to blockchain
        task_id = self._submit_task(
            project_id=project_id,
            spec_hash=bytes.fromhex(spec_hash),
            ground_truth_hash=bytes.fromhex(ground_truth_hash),
            learnability_reward=learnability_reward,
            solver_reward=solver_reward,
        )

        if task_id:
            self.pending_tasks[task_id] = task_spec
            logger.info(f"Task {task_id} created successfully")

        return task_id

    def _estimate_difficulty(self, input_data: Dict[str, Any]) -> float:
        """Estimate task difficulty (0-1 scale)."""
        # Simple heuristic based on data size
        data_size = len(json.dumps(input_data))
        return min(1.0, data_size / 10000)

    def _submit_task(
        self,
        project_id: int,
        spec_hash: bytes,
        ground_truth_hash: bytes,
        learnability_reward: int,
        solver_reward: int,
    ) -> Optional[int]:
        """Submit task to the blockchain."""
        # In production, this would call the TaskContract
        logger.info(f"Submitting task to blockchain...")

        # Mock: return a fake task ID
        import random
        return random.randint(1, 1000000)

    def reveal_ground_truth(self, task_id: int) -> bool:
        """
        Reveal the ground truth for a task after solutions are committed.

        Args:
            task_id: The task to reveal ground truth for

        Returns:
            True if successful
        """
        if task_id not in self.pending_tasks:
            logger.error(f"Task {task_id} not found in pending tasks")
            return False

        task = self.pending_tasks[task_id]
        logger.info(f"Revealing ground truth for task {task_id}")

        # In production, call ResultsRewards.revealGroundTruth
        logger.info(f"Ground truth CID: {task.ground_truth_cid}")

        del self.pending_tasks[task_id]
        return True


def main():
    """Run the proposer node."""
    node = ProposerNode()

    # Demo: generate a simple task
    task_id = node.generate_task(
        project_id=1,
        description="Train model to classify images",
        input_data={"images": ["img1.png", "img2.png"], "labels": [0, 1]},
        ground_truth={"accuracy": 0.95, "model_cid": "QmExample..."},
    )

    if task_id:
        print(f"Created task {task_id}")

    # Run for a few cycles
    node.run(max_cycles=3)


if __name__ == "__main__":
    main()
