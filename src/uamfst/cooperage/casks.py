from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, destination)


def read_json(path: str | Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(Path(path).read_text())
    return data
