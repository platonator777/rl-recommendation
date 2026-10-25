from __future__ import annotations

import hashlib


def user_hash_partition(uid: int | str, seed: int = 17, audit_fraction: float = 0.2) -> str:
    """Stable disjoint-user split that does not depend on Python's hash seed."""
    if not 0.0 < audit_fraction < 1.0:
        raise ValueError("audit_fraction must be between zero and one")
    digest = hashlib.sha256(f"{seed}:{uid}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return "audit" if bucket < audit_fraction else "fit"
