from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def manifest_sha256(entries: Mapping[str, object]) -> str:
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
