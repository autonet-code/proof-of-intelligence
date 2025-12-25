// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Votes.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";

/**
 * @title ATNToken
 * @dev The native ERC20 token for the Autonet ecosystem.
 *      Implements ERC20Votes for governance, ERC20Permit for gasless approvals, and ERC20Burnable.
 *      Used for: gas fees, staking, rewards, inference payments, governance.
 */
contract ATNToken is ERC20, ERC20Permit, ERC20Votes, ERC20Burnable {
    address public daoAddress;

    event DaoAddressChanged(address indexed newDaoAddress);

    /**
     * @param _initialHolder Address receiving initial supply
     * @param _initialSupply Total initial supply (without decimals)
     */
    constructor(
        address _initialHolder,
        uint256 _initialSupply
    )
        ERC20("Autonoma Token", "ATN")
        ERC20Permit("Autonoma Token")
    {
        require(_initialHolder != address(0), "ATNToken: zero initial holder");
        _mint(_initialHolder, _initialSupply * (10**decimals()));
        daoAddress = _initialHolder; // Initial holder is also initial DAO
    }

    /**
     * @dev Allows the DAO to mint new tokens.
     */
    function mint(address to, uint256 amount) external {
        require(msg.sender == daoAddress, "ATNToken: caller is not DAO");
        _mint(to, amount);
    }

    /**
     * @dev Allows the DAO to update its address.
     */
    function setDaoAddress(address _newDao) external {
        require(msg.sender == daoAddress, "ATNToken: caller is not current DAO");
        require(_newDao != address(0), "ATNToken: new DAO is zero");
        daoAddress = _newDao;
        emit DaoAddressChanged(_newDao);
    }

    // Required overrides for multiple inheritance
    function _update(address from, address to, uint256 amount) internal override(ERC20, ERC20Votes) {
        super._update(from, to, amount);
    }

    function nonces(address owner) public view override(ERC20Permit, Nonces) returns (uint256) {
        return super.nonces(owner);
    }
}
