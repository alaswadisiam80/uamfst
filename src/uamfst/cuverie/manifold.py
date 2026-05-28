from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar, cast

import gin

Configurable = TypeVar("Configurable", bound=Callable[..., object])
Result = TypeVar("Result")


def configurable(target: Configurable) -> Configurable:
    return cast(Configurable, gin.configurable(target))


def entrypoint(target: Callable[..., Result]) -> Callable[..., Result]:
    return cast(Callable[..., Result], gin.configurable(target))


def register(target: Callable[..., object], name: str) -> None:
    gin.external_configurable(target, name=name)


def ingest(paths: Sequence[str], bindings: Sequence[str] | None = None) -> None:
    gin.parse_config_files_and_bindings(list(paths), list(bindings or []))


def apply_bindings(bindings: Sequence[str]) -> None:
    gin.parse_config("\n".join(bindings))


def operative_config() -> str:
    return cast(str, gin.operative_config_str())


def clear() -> None:
    gin.clear_config()
