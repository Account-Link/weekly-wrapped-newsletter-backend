import base64
import os
from typing import Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def _ensure_key(key: str) -> bytes:
    # Accept hex or raw utf-8; expect 32 bytes.
    try:
        raw = bytes.fromhex(key)
    except ValueError:
        raw = key.encode()
    if len(raw) < 32:
        raw = raw.ljust(32, b"\0")
    return raw[:32]


def encrypt(token: str, secret: str) -> str:
    key = _ensure_key(secret)
    aesgcm = AESGCM(key)
    iv = os.urandom(12)
    ct = aesgcm.encrypt(iv, token.encode(), None)
    blob = iv + ct
    return base64.urlsafe_b64encode(blob).decode()


def decrypt(token_encrypted: str, secret: str) -> str:
    key = _ensure_key(secret)
    aesgcm = AESGCM(key)
    blob = base64.urlsafe_b64decode(token_encrypted.encode())
    iv, ct = blob[:12], blob[12:]
    return aesgcm.decrypt(iv, ct, None).decode()
