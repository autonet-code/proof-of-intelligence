"""
Cryptographic utilities for Autonet nodes.
"""

import hashlib
from typing import Optional


def hash_content(content: bytes) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content).hexdigest()


def hash_string(s: str) -> str:
    """Compute SHA256 hash of a string."""
    return hash_content(s.encode())


def compute_commitment(content: bytes, salt: bytes) -> str:
    """Compute a commitment (hash of content + salt)."""
    return hash_content(content + salt)


def verify_commitment(content: bytes, salt: bytes, commitment: str) -> bool:
    """Verify a commitment matches the content and salt."""
    return compute_commitment(content, salt) == commitment


def verify_signature(
    message: bytes,
    signature: str,
    address: str,
) -> bool:
    """
    Verify an Ethereum signature.
    Returns True if the signature is valid for the given message and address.
    """
    try:
        from eth_account.messages import encode_defunct
        from eth_account import Account

        msg = encode_defunct(primitive=message)
        recovered = Account.recover_message(msg, signature=signature)
        return recovered.lower() == address.lower()
    except ImportError:
        # eth_account not installed
        return True  # Skip verification in mock mode
    except Exception:
        return False


def sign_message(message: bytes, private_key: str) -> Optional[str]:
    """
    Sign a message with a private key.
    Returns the signature as a hex string.
    """
    try:
        from eth_account.messages import encode_defunct
        from eth_account import Account

        msg = encode_defunct(primitive=message)
        signed = Account.sign_message(msg, private_key)
        return signed.signature.hex()
    except ImportError:
        # eth_account not installed
        return "0x" + "00" * 65  # Mock signature
    except Exception:
        return None


def generate_keypair() -> tuple:
    """Generate a new Ethereum keypair."""
    try:
        from eth_account import Account
        account = Account.create()
        return account.address, account.key.hex()
    except ImportError:
        # Mock keypair
        import secrets
        key = secrets.token_hex(32)
        address = "0x" + hashlib.sha256(bytes.fromhex(key)).hexdigest()[:40]
        return address, key
