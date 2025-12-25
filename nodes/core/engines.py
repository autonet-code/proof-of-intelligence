"""
Autonet Node Engines

The four specialized engines that power each autonomous node:
- AwarenessEngine: Environmental perception
- GovernanceEngine: Constitutional validation and collective decision-making
- WorkEngine: Task execution (training, inference, verification)
- SurvivalEngine: Self-preservation and replication
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class InstructionStatus(Enum):
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"


@dataclass
class Instruction:
    id: str
    action: str
    details: Dict[str, Any]
    proof_of_adherence: str
    signature: str
    status: InstructionStatus = InstructionStatus.PENDING


class BaseEngine(ABC):
    """Base class for all node engines."""

    def __init__(self, node: "Node"):
        self.node = node
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{node.node_id[:8]}]")

    @abstractmethod
    def tick(self) -> None:
        """Execute one cycle of the engine."""
        pass


class AwarenessEngine(BaseEngine):
    """
    The node's duty to perceive its environment.
    Monitors network status, local resources, and external signals.
    """

    def __init__(self, node: "Node"):
        super().__init__(node)
        self.last_observation: Dict[str, Any] = {}

    def tick(self) -> None:
        self.last_observation = self.perceive()

    def perceive(self) -> Dict[str, Any]:
        """Gather environmental data."""
        import psutil

        observation = {
            "network_status": "OK",
            "cpu_percent": psutil.cpu_percent() if hasattr(psutil, 'cpu_percent') else 0.0,
            "memory_percent": psutil.virtual_memory().percent if hasattr(psutil, 'virtual_memory') else 0.0,
            "timestamp": self._current_time(),
            "consensus_alive": self.node.is_consensus_alive(),
        }

        self.logger.debug(f"Perception: {observation}")
        return observation

    def _current_time(self) -> float:
        import time
        return time.time()


class GovernanceEngine(BaseEngine):
    """
    The node's duty to participate in collective decision-making.
    Validates instructions against constitutional principles.
    """

    def __init__(self, node: "Node"):
        super().__init__(node)
        self.pending_instructions: List[Instruction] = []
        self.validated_instructions: List[Instruction] = []

    def tick(self) -> None:
        self.check_for_proposals()
        self.process_pending_instructions()

    def check_for_proposals(self) -> None:
        """Check for new proposals from the consensus network."""
        # In production, this would poll the blockchain or P2P network
        pass

    def process_pending_instructions(self) -> None:
        """Validate pending instructions against constitutional principles."""
        for instruction in list(self.pending_instructions):
            if self.validate_instruction(instruction):
                instruction.status = InstructionStatus.VALIDATED
                self.validated_instructions.append(instruction)
                self.node.work.queue_instruction(instruction)
            else:
                instruction.status = InstructionStatus.REJECTED
                self.logger.warning(f"Rejected instruction {instruction.id}: violates principles")

            self.pending_instructions.remove(instruction)

    def validate_instruction(self, instruction: Instruction) -> bool:
        """
        The node's "Right of Adherence" - validate against constitution.
        In production, this would use an LLM for semantic analysis.
        """
        return self.node.constitution.validate_action(
            instruction.action,
            instruction.proof_of_adherence
        )

    def submit_instruction(self, instruction: Instruction) -> None:
        """Add an instruction to the pending queue."""
        self.pending_instructions.append(instruction)


class WorkEngine(BaseEngine):
    """
    The node's duty to execute validated instructions.
    Handles training, inference, and verification tasks.
    """

    def __init__(self, node: "Node"):
        super().__init__(node)
        self.instruction_queue: List[Instruction] = []
        self.current_task: Optional[Instruction] = None

    def tick(self) -> None:
        if not self.node.is_consensus_alive():
            self.logger.warning("Consensus heartbeat missed. Halting work.")
            return

        self.execute_next()

    def queue_instruction(self, instruction: Instruction) -> None:
        """Add a validated instruction to the work queue."""
        self.instruction_queue.append(instruction)
        self.logger.info(f"Queued instruction: {instruction.id}")

    def execute_next(self) -> None:
        """Execute the next instruction in the queue."""
        if not self.instruction_queue:
            return

        instruction = self.instruction_queue.pop(0)
        self.current_task = instruction

        try:
            self._execute_instruction(instruction)
            instruction.status = InstructionStatus.EXECUTED
            self.logger.info(f"Executed: {instruction.action}")
        except Exception as e:
            instruction.status = InstructionStatus.FAILED
            self.logger.error(f"Failed to execute {instruction.id}: {e}")
        finally:
            self.current_task = None

    def _execute_instruction(self, instruction: Instruction) -> None:
        """Execute an instruction based on its action type."""
        action = instruction.action
        details = instruction.details

        if action == "TRAIN_MODEL":
            self._handle_training(details)
        elif action == "VERIFY_SOLUTION":
            self._handle_verification(details)
        elif action == "AGGREGATE_UPDATES":
            self._handle_aggregation(details)
        elif action == "SERVE_INFERENCE":
            self._handle_inference(details)
        else:
            self.logger.warning(f"Unknown action: {action}")

    def _handle_training(self, details: Dict[str, Any]) -> None:
        """Handle a training task."""
        self.logger.info(f"Training on task: {details.get('task_id')}")
        # In production: download model, train, upload update

    def _handle_verification(self, details: Dict[str, Any]) -> None:
        """Handle a verification task."""
        self.logger.info(f"Verifying solution: {details.get('solution_cid')}")
        # In production: download solution, verify against ground truth

    def _handle_aggregation(self, details: Dict[str, Any]) -> None:
        """Handle model aggregation."""
        self.logger.info(f"Aggregating updates for project: {details.get('project_id')}")
        # In production: download verified updates, perform FedAvg

    def _handle_inference(self, details: Dict[str, Any]) -> None:
        """Handle an inference request."""
        self.logger.info(f"Serving inference: {details.get('request_id')}")
        # In production: load model, run inference, return result


class SurvivalEngine(BaseEngine):
    """
    The node's duty to ensure its own persistence and spread.
    Handles replication and network maintenance.
    """

    def __init__(self, node: "Node"):
        super().__init__(node)
        self.replication_threshold = 0.8  # Replicate when network coverage < 80%

    def tick(self) -> None:
        self.maintain_presence()
        self.consider_replication()

    def maintain_presence(self) -> None:
        """Maintain node's presence in the network."""
        # In production: send heartbeat, update DHT, maintain connections
        pass

    def consider_replication(self) -> None:
        """Consider whether to spawn new nodes."""
        # In production: check network coverage, spawn spores if needed
        pass
