from __future__ import annotations


def interaction_ratio(
    full: float,
    without_first: float,
    without_second: float,
    without_both: float,
) -> float:
    combined = full - without_both
    individual = (full - without_first) + (full - without_second)
    if individual == 0.0:
        return float("nan")
    return combined / individual


def explanation_coverage(full: float, base: float, component_drops: list[float]) -> float:
    total = full - base
    if total == 0.0:
        return float("nan")
    return sum(component_drops) / total
