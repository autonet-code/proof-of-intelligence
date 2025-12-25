"""
Autonet Solver Node

Performs distributed AI training on assigned tasks.
Downloads global model, trains on local data, uploads model updates.

Implements Gensyn-style checkpoint generation for partial verification:
- Generates checkpoints at configurable intervals during training
- Each checkpoint includes weights hash, data indices, and deterministic seed
- Enables dispute resolution via checkpoint comparison
"""

import logging
import time
import json
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from ..core import Node, NodeRole, DEFAULT_CONSTITUTION
from ..common import BlockchainInterface, IPFSClient, hash_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingCheckpoint:
    """A checkpoint during training for Gensyn-style verification."""
    step_number: int
    weights_hash: str
    data_indices_hash: str
    random_seed: str
    timestamp: float
    weights_cid: Optional[str] = None  # IPFS CID of actual weights (optional)


@dataclass
class TrainingResult:
    """Result from a training run."""
    task_id: int
    model_update_cid: str
    metrics: Dict[str, float]
    training_time: float
    checkpoints: List[TrainingCheckpoint] = field(default_factory=list)
    checkpoint_frequency: int = 10  # Steps between checkpoints


class SolverNode(Node):
    """
    Solver node that performs distributed training.

    Responsibilities:
    - Download current global model from IPFS
    - Download task specifications
    - Perform local training with checkpoint generation
    - Upload model updates and checkpoints to IPFS
    - Commit solution hash to blockchain
    - Reveal solution after ground truth is revealed
    """

    def __init__(
        self,
        blockchain: Optional[BlockchainInterface] = None,
        ipfs: Optional[IPFSClient] = None,
        checkpoint_frequency: int = 10,
        deterministic_seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(role=NodeRole.SOLVER, **kwargs)
        self.blockchain = blockchain or BlockchainInterface()
        self.ipfs = ipfs or IPFSClient()
        self.completed_tasks: Dict[int, TrainingResult] = {}
        self.checkpoint_frequency = checkpoint_frequency
        self.deterministic_seed = deterministic_seed or int(time.time())

    def claim_task(self, task_id: int) -> bool:
        """
        Claim a task for training.

        Args:
            task_id: The task to claim

        Returns:
            True if successfully claimed
        """
        logger.info(f"Claiming task {task_id}")
        # In production, this would verify stake and call TaskContract
        return True

    def train(
        self,
        task_id: int,
        task_spec_cid: str,
        global_model_cid: Optional[str] = None,
    ) -> Optional[TrainingResult]:
        """
        Perform training on a task with checkpoint generation.

        Args:
            task_id: The task being trained
            task_spec_cid: IPFS CID of the task specification
            global_model_cid: IPFS CID of the current global model

        Returns:
            Training result if successful
        """
        logger.info(f"Starting training for task {task_id}")
        start_time = time.time()

        # Download task specification
        task_spec = self.ipfs.get_json(task_spec_cid)
        if not task_spec:
            logger.error(f"Failed to download task spec: {task_spec_cid}")
            return None

        # Download global model if provided
        if global_model_cid:
            model_data = self.ipfs.get_bytes(global_model_cid)
            if not model_data:
                logger.warning(f"Failed to download global model: {global_model_cid}")

        # Perform training with checkpointing
        logger.info("Training in progress with checkpointing...")
        model_update, checkpoints = self._train_model_with_checkpoints(task_spec, task_id)

        # Upload model update to IPFS
        update_cid = self.ipfs.add_json(model_update)
        if not update_cid:
            logger.error("Failed to upload model update")
            return None

        # Upload checkpoints to IPFS (optional, for detailed verification)
        for checkpoint in checkpoints:
            checkpoint_data = {
                "step_number": checkpoint.step_number,
                "weights_hash": checkpoint.weights_hash,
                "data_indices_hash": checkpoint.data_indices_hash,
                "random_seed": checkpoint.random_seed,
                "timestamp": checkpoint.timestamp,
            }
            checkpoint.weights_cid = self.ipfs.add_json(checkpoint_data)

        training_time = time.time() - start_time
        result = TrainingResult(
            task_id=task_id,
            model_update_cid=update_cid,
            metrics=model_update.get("metrics", {}),
            training_time=training_time,
            checkpoints=checkpoints,
            checkpoint_frequency=self.checkpoint_frequency,
        )

        self.completed_tasks[task_id] = result
        logger.info(f"Training completed for task {task_id} in {training_time:.2f}s")
        logger.info(f"Generated {len(checkpoints)} checkpoints")

        return result

    def _train_model_with_checkpoints(
        self, task_spec: Dict[str, Any], task_id: int
    ) -> tuple:
        """
        Perform actual model training with checkpoint generation.
        In production, this would use PyTorch/TensorFlow with RepOps.
        """
        checkpoints = []
        total_steps = 100  # Mock: 100 training steps

        # Initialize deterministic seed for reproducibility (RepOps style)
        current_seed = self._generate_deterministic_seed(task_id, 0)

        for step in range(total_steps):
            # Simulate training step
            time.sleep(0.01)  # Fast mock training

            # Generate checkpoint at configured frequency
            if step > 0 and step % self.checkpoint_frequency == 0:
                checkpoint = self._create_checkpoint(step, current_seed)
                checkpoints.append(checkpoint)
                logger.debug(f"Created checkpoint at step {step}")

            # Update seed for next step (deterministic progression)
            current_seed = self._generate_deterministic_seed(task_id, step + 1)

        # Final checkpoint
        final_checkpoint = self._create_checkpoint(total_steps, current_seed)
        checkpoints.append(final_checkpoint)

        # Mock training result
        model_update = {
            "model_weights": "mock_weights_base64",
            "metrics": {
                "loss": 0.15,
                "accuracy": 0.92,
            },
            "training_steps": total_steps,
            "checkpoint_frequency": self.checkpoint_frequency,
            "final_seed": current_seed,
            "repops_version": "1.0.0",  # RepOps version for reproducibility
        }

        return model_update, checkpoints

    def _create_checkpoint(self, step: int, seed: str) -> TrainingCheckpoint:
        """Create a training checkpoint with deterministic hashes."""
        # In production, these would be actual hashes of model weights and data indices
        weights_hash = hashlib.sha256(f"weights_step_{step}_{seed}".encode()).hexdigest()
        data_indices_hash = hashlib.sha256(f"data_indices_step_{step}_{seed}".encode()).hexdigest()

        return TrainingCheckpoint(
            step_number=step,
            weights_hash=weights_hash,
            data_indices_hash=data_indices_hash,
            random_seed=seed,
            timestamp=time.time(),
        )

    def _generate_deterministic_seed(self, task_id: int, step: int) -> str:
        """
        Generate a deterministic seed for reproducibility (RepOps-style).
        This ensures different solvers training the same task will get
        identical results if they follow the protocol correctly.
        """
        seed_input = f"{self.deterministic_seed}:{task_id}:{step}"
        return hashlib.sha256(seed_input.encode()).hexdigest()[:16]

    def _train_model(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Legacy: Perform actual model training without checkpoints.
        In production, this would use PyTorch/TensorFlow.
        """
        # Mock training result
        return {
            "model_weights": "mock_weights_base64",
            "metrics": {
                "loss": 0.15,
                "accuracy": 0.92,
            },
            "training_steps": 1000,
        }

    def commit_solution(self, task_id: int) -> bool:
        """
        Commit solution hash to the blockchain.

        Args:
            task_id: The task to commit solution for

        Returns:
            True if successful
        """
        if task_id not in self.completed_tasks:
            logger.error(f"No completed training for task {task_id}")
            return False

        result = self.completed_tasks[task_id]
        solution_hash = hash_string(result.model_update_cid)

        logger.info(f"Committing solution hash for task {task_id}")
        # In production, call TaskContract.commitSolution

        return True

    def submit_checkpoints(self, task_id: int) -> bool:
        """
        Submit checkpoints to the blockchain for verification.

        Args:
            task_id: The task to submit checkpoints for

        Returns:
            True if successful
        """
        if task_id not in self.completed_tasks:
            logger.error(f"No completed training for task {task_id}")
            return False

        result = self.completed_tasks[task_id]

        logger.info(f"Submitting {len(result.checkpoints)} checkpoints for task {task_id}")

        for checkpoint in result.checkpoints:
            # In production, call TaskContract.submitCheckpoint
            logger.debug(
                f"Checkpoint step {checkpoint.step_number}: "
                f"weights={checkpoint.weights_hash[:16]}..."
            )

        return True

    def reveal_solution(self, task_id: int) -> bool:
        """
        Reveal solution after ground truth is revealed.

        Args:
            task_id: The task to reveal solution for

        Returns:
            True if successful
        """
        if task_id not in self.completed_tasks:
            logger.error(f"No completed training for task {task_id}")
            return False

        result = self.completed_tasks[task_id]
        logger.info(f"Revealing solution for task {task_id}: {result.model_update_cid}")
        # In production, call ResultsRewards.revealSolution

        return True

    def get_checkpoint_proof(self, task_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the full checkpoint proof for a completed task.
        This can be used for dispute resolution.
        """
        if task_id not in self.completed_tasks:
            return None

        result = self.completed_tasks[task_id]
        return {
            "task_id": task_id,
            "solver": "self",  # Would be actual address
            "checkpoints": [
                {
                    "step_number": cp.step_number,
                    "weights_hash": cp.weights_hash,
                    "data_indices_hash": cp.data_indices_hash,
                    "random_seed": cp.random_seed,
                }
                for cp in result.checkpoints
            ],
            "checkpoint_frequency": result.checkpoint_frequency,
            "final_weights_hash": hash_string(result.model_update_cid),
        }


def main():
    """Run the solver node."""
    node = SolverNode(
        checkpoint_frequency=10,
        deterministic_seed=42,  # Fixed seed for reproducibility
    )

    # Demo: claim and train a task with checkpointing
    task_id = 1
    if node.claim_task(task_id):
        result = node.train(
            task_id=task_id,
            task_spec_cid="QmTaskSpec...",
            global_model_cid="QmGlobalModel...",
        )

        if result:
            node.submit_checkpoints(task_id)
            node.commit_solution(task_id)

            # Show checkpoint proof
            proof = node.get_checkpoint_proof(task_id)
            print(f"\nCheckpoint Proof:")
            print(f"  Total checkpoints: {len(proof['checkpoints'])}")
            print(f"  Checkpoint frequency: {proof['checkpoint_frequency']}")
            print(f"  Final hash: {proof['final_weights_hash'][:20]}...")

            print(f"\nTraining result: {result.metrics}")

    # Run for a few cycles
    node.run(max_cycles=3)


if __name__ == "__main__":
    main()
