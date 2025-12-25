// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/governance/utils/IVotes.sol";
import "../utils/AutonetLib.sol";

/**
 * @title DisputeManager
 * @dev Stake-weighted dispute resolution for checkpoint challenges and verification disputes.
 */
contract DisputeManager {
    IVotes public immutable votingToken;

    // Governance parameters (in basis points)
    uint256 public quorumNumerator = 2000;   // 20%
    uint256 public quorumDenominator = 10000;
    uint256 public supermajorityNum = 6600;  // 66%
    uint256 public supermajorityDen = 10000;

    uint256 public votingPeriod = 3 days;
    uint256 public nextDisputeId = 1;

    mapping(uint256 => AutonetLib.Dispute) public disputes;
    mapping(uint256 => mapping(address => bool)) public hasVoted;

    event DisputeCreated(uint256 indexed id, bytes32 contentHash, address indexed challenger);
    event Voted(uint256 indexed id, address indexed voter, bool supportInvalid, uint256 weight);
    event DisputeResolved(uint256 indexed id, bool invalidWins);

    constructor(address _votingToken) {
        votingToken = IVotes(_votingToken);
    }

    /**
     * @dev Create a new dispute.
     * @param contentHash Hash of the content being disputed (checkpoint root, verification, etc.)
     */
    function createDispute(bytes32 contentHash) external returns (uint256 disputeId) {
        disputeId = nextDisputeId++;

        disputes[disputeId] = AutonetLib.Dispute({
            contentHash: contentHash,
            snapshot: block.number,
            totalVotes: 0,
            invalidVotes: 0,
            status: AutonetLib.DisputeStatus.CREATED,
            createdAt: block.timestamp
        });

        emit DisputeCreated(disputeId, contentHash, msg.sender);
    }

    /**
     * @dev Vote on a dispute.
     * @param disputeId The dispute to vote on
     * @param supportInvalid True to vote that the content is invalid
     */
    function vote(uint256 disputeId, bool supportInvalid) external {
        AutonetLib.Dispute storage d = disputes[disputeId];
        require(d.status == AutonetLib.DisputeStatus.CREATED || d.status == AutonetLib.DisputeStatus.VOTING, "Not active");
        require(block.timestamp < d.createdAt + votingPeriod, "Voting ended");
        require(!hasVoted[disputeId][msg.sender], "Already voted");

        uint256 weight = votingToken.getPastVotes(msg.sender, d.snapshot);
        require(weight > 0, "No voting power");

        hasVoted[disputeId][msg.sender] = true;
        d.totalVotes += weight;
        if (supportInvalid) {
            d.invalidVotes += weight;
        }
        d.status = AutonetLib.DisputeStatus.VOTING;

        emit Voted(disputeId, msg.sender, supportInvalid, weight);

        // Check if quorum reached
        uint256 totalSupply = votingToken.getPastTotalSupply(d.snapshot);
        if (d.totalVotes * quorumDenominator >= totalSupply * quorumNumerator) {
            _resolveDispute(disputeId);
        }
    }

    /**
     * @dev Finalize a dispute after voting period ends.
     */
    function finalizeDispute(uint256 disputeId) external {
        AutonetLib.Dispute storage d = disputes[disputeId];
        require(d.status == AutonetLib.DisputeStatus.VOTING, "Not in voting");
        require(block.timestamp >= d.createdAt + votingPeriod, "Voting not ended");

        _resolveDispute(disputeId);
    }

    function _resolveDispute(uint256 disputeId) internal {
        AutonetLib.Dispute storage d = disputes[disputeId];

        bool invalidWins = d.invalidVotes * supermajorityDen >= d.totalVotes * supermajorityNum;
        d.status = invalidWins ? AutonetLib.DisputeStatus.RESOLVED_INVALID : AutonetLib.DisputeStatus.RESOLVED_VALID;

        emit DisputeResolved(disputeId, invalidWins);
    }

    function getDispute(uint256 disputeId) external view returns (AutonetLib.Dispute memory) {
        return disputes[disputeId];
    }
}
