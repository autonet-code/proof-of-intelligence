"""
Autonet Node

The sovereign citizen-cell of the Autonet organism.
Each node operates according to constitutional principles and participates
in distributed AI training, verification, and governance.
"""

import uuid
import time
import logging
from typing import Optional
from enum import Enum

from .constitution import Constitution, DEFAULT_CONSTITUTION
from .engines import AwarenessEngine, GovernanceEngine, WorkEngine, SurvivalEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class NodeRole(Enum):
    """The role a node plays in the network."""
    PROPOSER = "proposer"
    SOLVER = "solver"
    COORDINATOR = "coordinator"
    AGGREGATOR = "aggregator"
    VALIDATOR = "validator"
    FULL = "full"  # Participates in all roles


class Node:
    """
    The sovereign citizen-cell of the Autonet organism.

    Each node:
    - Operates according to immutable constitutional principles
    - Participates in distributed AI training and verification
    - Validates instructions before execution
    - Only works when consensus heartbeat is active
    """

    def __init__(
        self,
        constitution: Constitution = DEFAULT_CONSTITUTION,
        role: NodeRole = NodeRole.FULL,
        node_id: Optional[str] = None,
    ):
        self.node_id = node_id or f"node_{uuid.uuid4().hex}"
        self.constitution = constitution
        self.role = role
        self.last_heartbeat: Optional[float] = None
        self.running = False

        # Initialize engines
        self.awareness = AwarenessEngine(self)
        self.governance = GovernanceEngine(self)
        self.work = WorkEngine(self)
        self.survival = SurvivalEngine(self)

        logger.info(
            f"[{self.node_id[:12]}] Node born with role={role.value}, "
            f"upholding {len(constitution.principles)} principles"
        )

    def run(self, max_cycles: Optional[int] = None) -> None:
        """Run the node's main lifecycle."""
        self.running = True
        cycle = 0

        logger.info(f"[{self.node_id[:12]}] Starting lifecycle...")

        try:
            while self.running:
                self._run_cycle()
                cycle += 1

                if max_cycles and cycle >= max_cycles:
                    logger.info(f"[{self.node_id[:12]}] Completed {max_cycles} cycles")
                    break

                time.sleep(self.constitution.operational_blueprint.get(
                    "heartbeat_interval_seconds", 60
                ) / 6)  # Run 6 times per heartbeat interval

        except KeyboardInterrupt:
            logger.info(f"[{self.node_id[:12]}] Received shutdown signal")
        finally:
            self.running = False

    def _run_cycle(self) -> None:
        """Execute one lifecycle cycle."""
        # 1. Perceive environment
        self.awareness.tick()

        # 2. Participate in governance
        self.governance.tick()

        # 3. Execute work if consensus is alive
        if self.is_consensus_alive():
            self.work.tick()
        else:
            logger.warning(f"[{self.node_id[:12]}] Consensus missed - halting work")

        # 4. Maintain presence and consider replication
        self.survival.tick()

    def is_consensus_alive(self) -> bool:
        """Check if the consensus network heartbeat is active."""
        heartbeat_interval = self.constitution.operational_blueprint.get(
            "heartbeat_interval_seconds", 60
        )

        if self.last_heartbeat is None:
            self.last_heartbeat = time.time()
            return True

        return (time.time() - self.last_heartbeat) <= heartbeat_interval

    def receive_heartbeat(self) -> None:
        """Record receipt of consensus heartbeat."""
        self.last_heartbeat = time.time()
        logger.debug(f"[{self.node_id[:12]}] Heartbeat received")

    def stop(self) -> None:
        """Stop the node's lifecycle."""
        self.running = False
        logger.info(f"[{self.node_id[:12]}] Stopping...")

    def get_status(self) -> dict:
        """Get current node status."""
        return {
            "node_id": self.node_id,
            "role": self.role.value,
            "running": self.running,
            "consensus_alive": self.is_consensus_alive(),
            "pending_instructions": len(self.governance.pending_instructions),
            "work_queue": len(self.work.instruction_queue),
            "last_observation": self.awareness.last_observation,
        }


def create_node(
    role: str = "full",
    constitution: Optional[Constitution] = None,
) -> Node:
    """Factory function to create a new node."""
    role_enum = NodeRole(role.lower())
    const = constitution or DEFAULT_CONSTITUTION
    return Node(constitution=const, role=role_enum)
