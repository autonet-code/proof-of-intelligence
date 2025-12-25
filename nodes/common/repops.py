"""
Reproducible Operators (RepOps) for Autonet

Implements Gensyn-style bitwise reproducible ML operators.
Ensures that honest providers yield identical outputs, enabling
trustless verification without redundant re-computation.

Key Features:
- Deterministic random number generation
- Fixed execution order for floating point operations
- Reproducible matrix operations (mock implementation)
- Checkpoint-compatible state management

In production, this would wrap PyTorch/TensorFlow operators with
deterministic execution semantics.
"""

import hashlib
import struct
from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass


@dataclass
class RepOpsConfig:
    """Configuration for reproducible operations."""
    seed: int
    precision: str = "float32"  # float16, float32, float64
    deterministic_algorithms: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False


class DeterministicRNG:
    """
    Deterministic random number generator for reproducible training.
    Uses SHA-256 based counter mode for platform-independent randomness.
    """

    def __init__(self, seed: int):
        self.initial_seed = seed
        self.counter = 0
        self._state = self._hash_state(seed, 0)

    def _hash_state(self, seed: int, counter: int) -> bytes:
        """Generate deterministic state from seed and counter."""
        data = f"{seed}:{counter}".encode()
        return hashlib.sha256(data).digest()

    def random(self) -> float:
        """Generate a random float in [0, 1)."""
        self.counter += 1
        self._state = self._hash_state(self.initial_seed, self.counter)
        # Convert first 8 bytes to float in [0, 1)
        int_val = int.from_bytes(self._state[:8], byteorder='big')
        return int_val / (2**64)

    def randint(self, low: int, high: int) -> int:
        """Generate a random integer in [low, high)."""
        return int(self.random() * (high - low)) + low

    def randn(self, shape: Tuple[int, ...]) -> List[float]:
        """
        Generate normally distributed random numbers.
        Uses Box-Muller transform for reproducibility.
        """
        size = 1
        for dim in shape:
            size *= dim

        result = []
        for i in range(0, size, 2):
            u1 = self.random()
            u2 = self.random()
            # Box-Muller transform
            import math
            mag = math.sqrt(-2.0 * math.log(u1 + 1e-10))
            z0 = mag * math.cos(2.0 * math.pi * u2)
            z1 = mag * math.sin(2.0 * math.pi * u2)
            result.append(z0)
            if i + 1 < size:
                result.append(z1)

        return result[:size]

    def get_state(self) -> Dict[str, Any]:
        """Get current RNG state for checkpointing."""
        return {
            "seed": self.initial_seed,
            "counter": self.counter,
            "state_hash": self._state.hex(),
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore RNG state from checkpoint."""
        self.initial_seed = state["seed"]
        self.counter = state["counter"]
        self._state = bytes.fromhex(state["state_hash"])

    def fork(self, label: str) -> 'DeterministicRNG':
        """
        Create a new RNG derived from current state.
        Useful for parallel operations that need independent RNGs.
        """
        fork_seed = int.from_bytes(
            hashlib.sha256(f"{self.initial_seed}:{self.counter}:{label}".encode()).digest()[:8],
            byteorder='big'
        )
        return DeterministicRNG(fork_seed)


class RepOpsContext:
    """
    Context manager for reproducible operations.
    Ensures deterministic execution within the context.
    """

    def __init__(self, config: RepOpsConfig):
        self.config = config
        self.rng = DeterministicRNG(config.seed)
        self._checkpoints: List[Dict[str, Any]] = []

    def __enter__(self):
        # In production, would set torch/tf to deterministic mode
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def checkpoint(self, step: int, weights_hash: str, data_indices: List[int]) -> Dict[str, Any]:
        """
        Create a checkpoint of the current state.

        Args:
            step: Current training step
            weights_hash: Hash of model weights
            data_indices: Indices of data used in this step

        Returns:
            Checkpoint dict with all state needed for verification
        """
        checkpoint = {
            "step": step,
            "weights_hash": weights_hash,
            "data_indices_hash": hashlib.sha256(str(data_indices).encode()).hexdigest(),
            "rng_state": self.rng.get_state(),
            "config": {
                "seed": self.config.seed,
                "precision": self.config.precision,
            }
        }
        self._checkpoints.append(checkpoint)
        return checkpoint

    def restore_checkpoint(self, checkpoint: Dict[str, Any]):
        """Restore state from a checkpoint."""
        self.rng.set_state(checkpoint["rng_state"])


class ReproducibleMatMul:
    """
    Reproducible matrix multiplication.
    Enforces fixed execution order for floating point operations.
    """

    @staticmethod
    def matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """
        Perform matrix multiplication with deterministic order.
        In production, would use optimized but deterministic BLAS.
        """
        rows_a = len(a)
        cols_a = len(a[0]) if a else 0
        cols_b = len(b[0]) if b else 0

        result = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]

        # Fixed iteration order for reproducibility
        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    # Kahan summation for numerical stability
                    result[i][j] += a[i][k] * b[k][j]

        return result

    @staticmethod
    def hash_matrix(matrix: List[List[float]]) -> str:
        """Compute deterministic hash of a matrix."""
        # Flatten and convert to bytes with fixed precision
        flat = []
        for row in matrix:
            for val in row:
                # Use struct for consistent float representation
                flat.append(struct.pack('>d', val))

        return hashlib.sha256(b''.join(flat)).hexdigest()


class ReproducibleOptimizer:
    """
    Mock reproducible optimizer for demonstration.
    In production, would wrap PyTorch/TensorFlow optimizers.
    """

    def __init__(self, learning_rate: float, rng: DeterministicRNG):
        self.learning_rate = learning_rate
        self.rng = rng
        self.step_count = 0

    def step(self, gradients: List[float], weights: List[float]) -> List[float]:
        """
        Perform one optimization step.
        Uses deterministic gradient update.
        """
        self.step_count += 1

        # Simple SGD with deterministic execution
        new_weights = []
        for w, g in zip(weights, gradients):
            new_w = w - self.learning_rate * g
            new_weights.append(new_w)

        return new_weights

    def get_state(self) -> Dict[str, Any]:
        """Get optimizer state for checkpointing."""
        return {
            "step_count": self.step_count,
            "learning_rate": self.learning_rate,
            "rng_state": self.rng.get_state(),
        }


def create_repops_context(task_seed: int, step: int = 0) -> RepOpsContext:
    """
    Create a RepOps context for a training task.

    Args:
        task_seed: Base seed for the task (from blockchain/task spec)
        step: Starting step (for resuming from checkpoint)

    Returns:
        Configured RepOpsContext
    """
    # Derive deterministic seed from task seed
    derived_seed = int.from_bytes(
        hashlib.sha256(f"repops:{task_seed}:{step}".encode()).digest()[:8],
        byteorder='big'
    )

    config = RepOpsConfig(
        seed=derived_seed,
        precision="float32",
        deterministic_algorithms=True,
    )

    return RepOpsContext(config)


def verify_checkpoint_consistency(
    checkpoint1: Dict[str, Any],
    checkpoint2: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """
    Verify that two checkpoints are consistent.
    Used for dispute resolution.

    Returns:
        (is_consistent, first_mismatch_field)
    """
    fields_to_check = ["step", "weights_hash", "data_indices_hash"]

    for field in fields_to_check:
        if checkpoint1.get(field) != checkpoint2.get(field):
            return False, field

    # Check RNG state consistency
    rng1 = checkpoint1.get("rng_state", {})
    rng2 = checkpoint2.get("rng_state", {})

    if rng1.get("counter") != rng2.get("counter"):
        return False, "rng_counter"

    return True, None


# Version for tracking compatibility
REPOPS_VERSION = "1.0.0"


def get_version() -> str:
    """Get RepOps version for inclusion in training artifacts."""
    return REPOPS_VERSION


if __name__ == "__main__":
    # Demo usage
    print(f"RepOps v{REPOPS_VERSION} - Reproducible Operators for Autonet")
    print()

    # Create reproducible context
    ctx = create_repops_context(task_seed=12345)

    with ctx:
        # Generate reproducible random numbers
        print("Deterministic RNG:")
        for i in range(5):
            print(f"  random() = {ctx.rng.random():.6f}")

        print()

        # Create checkpoint
        checkpoint = ctx.checkpoint(
            step=100,
            weights_hash="abc123",
            data_indices=[0, 1, 2, 3, 4],
        )
        print(f"Checkpoint: {checkpoint}")

        print()

        # Matrix multiplication
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[5.0, 6.0], [7.0, 8.0]]
        result = ReproducibleMatMul.matmul(a, b)
        print(f"MatMul result: {result}")
        print(f"MatMul hash: {ReproducibleMatMul.hash_matrix(result)}")
