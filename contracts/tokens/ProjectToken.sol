// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title ProjectToken
 * @dev Project-specific ERC20 tokens (PTs) for AI development projects.
 *      Represents shares in a project and grants inference discounts.
 *      The deploying Project contract is the owner with mint/burn rights.
 */
contract ProjectToken is ERC20, ERC20Burnable, Ownable {
    constructor(
        string memory _name,
        string memory _symbol,
        address _owner
    ) ERC20(_name, _symbol) Ownable(_owner) {}

    /**
     * @dev Mints tokens to an address. Only callable by owner (Project contract).
     */
    function mint(address to, uint256 amount) external onlyOwner {
        _mint(to, amount);
    }

    /**
     * @dev Burns tokens from an address. Only callable by owner.
     */
    function burnFrom(address from, uint256 amount) public override onlyOwner {
        _burn(from, amount);
    }
}
