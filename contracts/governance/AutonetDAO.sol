// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/governance/utils/IVotes.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title AutonetDAO
 * @dev Core governance contract for the Autonet ecosystem.
 *      Manages parameter updates, treasury, and protocol upgrades.
 */
contract AutonetDAO is Ownable {
    IVotes public votingToken;
    address public treasury;

    // Governance parameters
    uint256 public proposalThreshold = 1000 * 1e18; // Min tokens to propose
    uint256 public votingDelay = 1 days;
    uint256 public votingPeriod = 7 days;
    uint256 public quorumVotes = 100000 * 1e18; // Min votes for quorum

    uint256 public nextProposalId = 1;

    enum ProposalState { Pending, Active, Canceled, Defeated, Succeeded, Queued, Expired, Executed }

    struct Proposal {
        uint256 id;
        address proposer;
        string description;
        uint256 startBlock;
        uint256 endBlock;
        uint256 forVotes;
        uint256 againstVotes;
        bool executed;
        bool canceled;
        bytes32 calldataHash;
    }

    mapping(uint256 => Proposal) public proposals;
    mapping(uint256 => mapping(address => bool)) public hasVoted;

    // Authorized contracts that can be called by governance
    mapping(address => bool) public authorizedTargets;

    event ProposalCreated(uint256 indexed proposalId, address indexed proposer, string description);
    event VoteCast(uint256 indexed proposalId, address indexed voter, bool support, uint256 weight);
    event ProposalExecuted(uint256 indexed proposalId);
    event ProposalCanceled(uint256 indexed proposalId);
    event TreasuryUpdated(address indexed newTreasury);
    event TargetAuthorized(address indexed target, bool authorized);

    constructor(address _votingToken, address _treasury) Ownable(msg.sender) {
        votingToken = IVotes(_votingToken);
        treasury = _treasury;
    }

    function setTreasury(address _treasury) external onlyOwner {
        treasury = _treasury;
        emit TreasuryUpdated(_treasury);
    }

    function setAuthorizedTarget(address _target, bool _authorized) external onlyOwner {
        authorizedTargets[_target] = _authorized;
        emit TargetAuthorized(_target, _authorized);
    }

    function propose(string calldata description, bytes32 calldataHash) external returns (uint256) {
        require(votingToken.getVotes(msg.sender) >= proposalThreshold, "Below threshold");

        uint256 proposalId = nextProposalId++;
        uint256 startBlock = block.number + votingDelay / 12; // ~12s blocks
        uint256 endBlock = startBlock + votingPeriod / 12;

        proposals[proposalId] = Proposal({
            id: proposalId,
            proposer: msg.sender,
            description: description,
            startBlock: startBlock,
            endBlock: endBlock,
            forVotes: 0,
            againstVotes: 0,
            executed: false,
            canceled: false,
            calldataHash: calldataHash
        });

        emit ProposalCreated(proposalId, msg.sender, description);
        return proposalId;
    }

    function castVote(uint256 proposalId, bool support) external {
        Proposal storage p = proposals[proposalId];
        require(block.number >= p.startBlock, "Voting not started");
        require(block.number <= p.endBlock, "Voting ended");
        require(!hasVoted[proposalId][msg.sender], "Already voted");

        uint256 weight = votingToken.getPastVotes(msg.sender, p.startBlock);
        require(weight > 0, "No voting power");

        hasVoted[proposalId][msg.sender] = true;
        if (support) {
            p.forVotes += weight;
        } else {
            p.againstVotes += weight;
        }

        emit VoteCast(proposalId, msg.sender, support, weight);
    }

    function execute(uint256 proposalId, address target, bytes calldata data) external {
        Proposal storage p = proposals[proposalId];
        require(block.number > p.endBlock, "Voting not ended");
        require(!p.executed, "Already executed");
        require(!p.canceled, "Canceled");
        require(p.forVotes > p.againstVotes, "Not passed");
        require(p.forVotes >= quorumVotes, "Quorum not reached");
        require(keccak256(data) == p.calldataHash, "Calldata mismatch");
        require(authorizedTargets[target], "Target not authorized");

        p.executed = true;

        (bool success,) = target.call(data);
        require(success, "Execution failed");

        emit ProposalExecuted(proposalId);
    }

    function cancel(uint256 proposalId) external {
        Proposal storage p = proposals[proposalId];
        require(msg.sender == p.proposer || msg.sender == owner(), "Not authorized");
        require(!p.executed, "Already executed");

        p.canceled = true;
        emit ProposalCanceled(proposalId);
    }

    function getProposalState(uint256 proposalId) external view returns (ProposalState) {
        Proposal storage p = proposals[proposalId];
        if (p.canceled) return ProposalState.Canceled;
        if (p.executed) return ProposalState.Executed;
        if (block.number < p.startBlock) return ProposalState.Pending;
        if (block.number <= p.endBlock) return ProposalState.Active;
        if (p.forVotes <= p.againstVotes || p.forVotes < quorumVotes) return ProposalState.Defeated;
        return ProposalState.Succeeded;
    }
}
