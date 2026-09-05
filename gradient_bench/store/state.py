"""state.py - the in-progress loop, mirrored to disk on every step.

The chemist starts a run, closes the laptop, and comes back tomorrow. Nothing
about that should lose the suggested method or the trace they already uploaded.
Every mutation writes state.json; opening the campaign reads it back.

The ANALYSIS is deliberately not serialised - it is recomputed by re-picking,
which is cheap and guarantees the numbers on screen came from the code that is
running now rather than from a stale cache.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class PendingRun:
    kind: str = "design"                    # "design" | "reference"
    u: Optional[list[float]] = None         # the method on the instrument
    phase: str = ""                         # cold_start | optimise | anchor | reference
    reason: str = ""
    seed_index: Optional[int] = None        # which cold-start seed this is
    n_replicates: int = 2
    uploads: list[str] = field(default_factory=list)   # trace paths, in run order
    picker_overrides: dict[str, Any] = field(default_factory=dict)
    manual_humps: list[list[float]] = field(default_factory=list)
    picked: bool = False
    posterior_mean: Optional[float] = None
    posterior_sd: Optional[float] = None
    acq_value: Optional[float] = None

    @property
    def ready_to_pick(self) -> bool:
        return len(self.uploads) >= max(1, int(self.n_replicates))

    def step(self) -> str:
        if self.u is None:
            return "suggest"
        if not self.ready_to_pick:
            return "upload"
        if not self.picked:
            return "pick"
        return "record"


def load(path: str) -> Optional[PendingRun]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            raw = json.load(f)
        if not raw:
            return None
        p = PendingRun(**{k: v for k, v in raw.items()
                          if k in PendingRun.__dataclass_fields__})
        p.uploads = [u for u in p.uploads if os.path.exists(u)]
        return p
    except Exception:
        return None


def save(path: str, pending: Optional[PendingRun]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(asdict(pending) if pending else None, f, indent=2)
    os.replace(tmp, path)


def clear(path: str) -> None:
    save(path, None)
