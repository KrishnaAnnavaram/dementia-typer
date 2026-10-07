"""Runtime settings from environment variables, with safe defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be {minimum} or more, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    output_dir: Path
    seed: int
    join_window_days: int
    n_splits: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(os.environ.get("DEMENTIA_DATA_DIR", "").strip() or "data/oasis3"),
            output_dir=Path(os.environ.get("DEMENTIA_OUTPUT_DIR", "").strip() or "runs"),
            seed=_int("DEMENTIA_SEED", 42, 0),
            join_window_days=_int("DEMENTIA_JOIN_WINDOW_DAYS", 365, 0),
            n_splits=_int("DEMENTIA_N_SPLITS", 5, 2),
        )
