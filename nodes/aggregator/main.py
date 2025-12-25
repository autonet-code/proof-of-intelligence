"""
Autonet Aggregator Node

Combines verified model updates into improved global models.
Implements federated averaging (FedAvg) and other aggregation strategies.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..core import Node, NodeRole, DEFAULT_CONSTITUTION
from ..common import BlockchainInterface, IPFSClient, hash_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """Result of model aggregation."""
    project_id: int
    round_number: int
    new_model_cid: str
    updates_aggregated: int
    metrics: Dict[str, float]


class AggregatorNode(Node):
    """
    Aggregator node that combines verified model updates.

    Responsibilities:
    - Collect verified model updates from IPFS
    - Perform federated averaging
    - Upload new global model to IPFS
    - Update project contract with new model CID
    """

    def __init__(
        self,
        blockchain: Optional[BlockchainInterface] = None,
        ipfs: Optional[IPFSClient] = None,
        **kwargs,
    ):
        super().__init__(role=NodeRole.AGGREGATOR, **kwargs)
        self.blockchain = blockchain or BlockchainInterface()
        self.ipfs = ipfs or IPFSClient()
        self.aggregation_history: List[AggregationResult] = []

    def aggregate_updates(
        self,
        project_id: int,
        round_number: int,
        update_cids: List[str],
        current_model_cid: Optional[str] = None,
    ) -> Optional[AggregationResult]:
        """
        Aggregate multiple model updates into a new global model.

        Args:
            project_id: The project being updated
            round_number: Current training round
            update_cids: List of IPFS CIDs for verified model updates
            current_model_cid: CID of the current global model

        Returns:
            Aggregation result if successful
        """
        logger.info(f"Aggregating {len(update_cids)} updates for project {project_id}")

        if len(update_cids) == 0:
            logger.warning("No updates to aggregate")
            return None

        # Download all updates
        updates = []
        for cid in update_cids:
            update = self.ipfs.get_json(cid)
            if update:
                updates.append(update)
            else:
                logger.warning(f"Failed to download update: {cid}")

        if len(updates) == 0:
            logger.error("Failed to download any updates")
            return None

        # Download current model if provided
        current_model = None
        if current_model_cid:
            current_model = self.ipfs.get_json(current_model_cid)

        # Perform aggregation
        new_model = self._fedavg(updates, current_model)

        # Compute aggregate metrics
        metrics = self._compute_aggregate_metrics(updates)

        # Add metadata
        new_model["metadata"] = {
            "project_id": project_id,
            "round": round_number,
            "updates_count": len(updates),
            "metrics": metrics,
        }

        # Upload new model to IPFS
        new_model_cid = self.ipfs.add_json(new_model)
        if not new_model_cid:
            logger.error("Failed to upload aggregated model")
            return None

        result = AggregationResult(
            project_id=project_id,
            round_number=round_number,
            new_model_cid=new_model_cid,
            updates_aggregated=len(updates),
            metrics=metrics,
        )

        self.aggregation_history.append(result)
        logger.info(f"Aggregation complete: new model CID = {new_model_cid[:20]}...")

        return result

    def _fedavg(
        self,
        updates: List[Dict[str, Any]],
        current_model: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform Federated Averaging on model updates.

        In production, this would:
        - Parse model weights from each update
        - Compute weighted average based on training samples
        - Return the averaged weights
        """
        # Mock implementation
        logger.info(f"Performing FedAvg on {len(updates)} updates")

        # For demo, just return a merged structure
        aggregated = {
            "model_weights": "aggregated_weights_base64",
            "aggregation_method": "fedavg",
            "num_updates": len(updates),
        }

        if current_model:
            aggregated["previous_model"] = current_model.get("model_weights", "none")

        return aggregated

    def _compute_aggregate_metrics(
        self,
        updates: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute aggregate metrics across all updates."""
        if not updates:
            return {}

        # Collect all metrics
        all_metrics = [u.get("metrics", {}) for u in updates]

        # Compute averages
        result = {}
        metric_keys = set()
        for m in all_metrics:
            metric_keys.update(m.keys())

        for key in metric_keys:
            values = [m.get(key, 0) for m in all_metrics if key in m]
            if values:
                result[f"avg_{key}"] = sum(values) / len(values)
                result[f"min_{key}"] = min(values)
                result[f"max_{key}"] = max(values)

        return result

    def update_project_model(self, project_id: int, new_model_cid: str) -> bool:
        """
        Update the project contract with the new global model.

        Args:
            project_id: The project to update
            new_model_cid: CID of the new global model

        Returns:
            True if successful
        """
        logger.info(f"Updating project {project_id} model to {new_model_cid[:20]}...")

        # In production, call Project.setMatureModel or similar

        return True


def main():
    """Run the aggregator node."""
    node = AggregatorNode()

    # Demo: aggregate some updates
    result = node.aggregate_updates(
        project_id=1,
        round_number=1,
        update_cids=["QmUpdate1...", "QmUpdate2...", "QmUpdate3..."],
        current_model_cid="QmCurrentModel...",
    )

    if result:
        node.update_project_model(result.project_id, result.new_model_cid)
        print(f"Aggregation result: {result}")

    # Run for a few cycles
    node.run(max_cycles=3)


if __name__ == "__main__":
    main()
