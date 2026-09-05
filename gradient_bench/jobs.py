"""jobs.py - background work, so a fit never freezes the page.

Fitting the GP and optimising the acquisition runs into seconds. In a request
cycle that is a dead button and a user who clicks again. Long calls become jobs
the page polls, with a real state instead of a spinner that means nothing.

Deliberately in-process and single-machine: this app is one chemist at one
bench beside one instrument, and a job queue that needs a broker would be
infrastructure serving nobody.
"""
from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

_LOCK = threading.Lock()
_JOBS: dict[str, "Job"] = {}
MAX_KEPT = 200


@dataclass
class Job:
    id: str
    label: str
    state: str = "running"          # running | done | error
    progress: str = ""
    result: Any = None
    error: Optional[str] = None
    detail: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "state": self.state,
                "progress": self.progress, "result": self.result,
                "error": self.error}


def submit(label: str, fn: Callable[[Callable[[str], None]], Any]) -> Job:
    """Run fn in a thread. fn receives a `progress(text)` callback."""
    job = Job(id=uuid.uuid4().hex[:12], label=label)
    with _LOCK:
        _JOBS[job.id] = job
        if len(_JOBS) > MAX_KEPT:
            for k in list(_JOBS)[:len(_JOBS) - MAX_KEPT]:
                if _JOBS[k].state != "running":
                    _JOBS.pop(k, None)

    def progress(text: str) -> None:
        job.progress = text

    def run() -> None:
        try:
            job.result = fn(progress)
            job.state = "done"
            job.progress = ""
        except Exception as exc:
            job.state = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.detail = traceback.format_exc()

    threading.Thread(target=run, daemon=True).start()
    return job


def get(job_id: str) -> Optional[Job]:
    with _LOCK:
        return _JOBS.get(job_id)
