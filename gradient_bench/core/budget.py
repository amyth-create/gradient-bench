"""budget.py - when a campaign stops, and what it costs to keep going.

TWO EXITS, ONE HARD AND ONE HUMAN.  A campaign ends when the run budget is
used up, or when the analyst judges the separation good enough and closes it.
Both are real endings and the sheet records which one happened. There is no
automatic convergence test: the honest stopping signal - a flat uncertainty
trend with the reference still passing - is a judgement made by looking, and
the analyst is the one who looks.

THE BUDGET COUNTS UNIQUE METHODS.  One distinct set of parameters costs one,
however many times it is injected. Replicates cost nothing extra and neither do
instrument checks.

This reverses what this module originally did, and the reasoning it was
reversed on is worth keeping. Counting injections made the budget a direct
measure of bench time, which is the honest thing to ration - but with
replicates mandatory it also meant the number an analyst sets is not the number
they are planning with. A forty-injection budget at two repeats buys about
fifteen methods, and "how many methods do I get to try" is the question being
asked. Amyth's call, 2026-08-30: the budget counts methods.

The cost of that choice, stated rather than hidden: bench time now scales with
the replicate count instead of being fixed by the budget. Forty methods is
about 89 injections at two repeats and about 129 at three. So every screen that
sets or reports a budget also states the injections it implies - see
`injection_cost`. A budget that quietly means two and a half times the
instrument time it used to is the failure mode this replaces, and the only
defence against it is arithmetic in front of the analyst.

EXHAUSTION IS NOT A HALT.  A HALT means the instrument is untrustworthy and the
numbers coming off it may be wrong. Running out of budget means the plan is
spent; every number already recorded is exactly as good as it was. So this
blocks PROPOSING and nothing else - recording an in-flight run, reading,
charting and exporting all continue to work. Conflating the two would teach
the analyst to ignore both.

A RUN THAT CANNOT BE FINISHED IS WORSE THAN NO RUN.  `record` refuses a method
with fewer traces than its replicate count, because a lone observation at a new
method leaves the noise term unidentifiable. Under a method budget this is no
longer a budget problem - a method costs one whether it takes two injections or
three, so a budget with one left can always serve the next method whole. The
`need` field is kept because the question "does the next thing fit" is still
the right one to ask, and because a future unit change must not silently
resurrect the partial-method bug this was written for.

EXTENDING IS ALLOWED, EXPLICITLY AND ON THE RECORD.  The analyst may extend;
nothing extends itself and nothing raises the ceiling silently. Every extension
appends to `budget_history` with its size, the position it happened at, and a
reason, because a campaign that ran forty methods straight and one that reached
forty through three extensions are different experiments and a reader six
months later cannot otherwise tell them apart. The reason is required for that same
purpose: an unexplained extension is precisely what the history exists to
prevent.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Optional

#: UNIQUE METHODS, not injections. Per-campaign; a scouting campaign may want
#: far fewer. At two repeats this is about 89 injections and roughly 45 hours
#: on a 30-minute gradient, which `injection_cost` states wherever it is set.
DEFAULT_RUN_BUDGET = 40

#: Warn once this many methods or fewer are left. Now in the budget's own
#: units, so it no longer has to be converted through the replicate count.
WARN_METHODS = 3

#: Proposals between scheduled instrument checks. Mirrors `drift.REFERENCE_EVERY`
#: and is used only to ESTIMATE the injection cost of a budget - drift.py
#: remains the authority on when a reference is actually due.
REFERENCE_EVERY_ESTIMATE = 5


def injection_cost(n_methods: int, n_replicates: int,
                   minutes_per_run: int = 30) -> dict[str, int]:
    """What a method budget costs on the instrument.

    A method budget is easier to plan with and hides the thing that is actually
    scarce, so the arithmetic is done here once and shown everywhere the budget
    appears rather than left for the analyst to do in their head.

    The anchor is the +1: one instrument check before anything else, then one
    every `REFERENCE_EVERY_ESTIMATE` methods. An ESTIMATE, deliberately - the
    real schedule reacts to what the reference does.
    """
    methods = max(0, int(n_methods))
    reps = max(1, int(n_replicates))
    design = methods * reps
    checks = 1 + methods // max(1, REFERENCE_EVERY_ESTIMATE) if methods else 0
    total = design + checks
    return {"methods": methods, "n_replicates": reps,
            "design_injections": design, "instrument_checks": checks,
            "injections": total, "hours": round(total * minutes_per_run / 60)}

EVENT_CREATED = "created"
EVENT_EXTENDED = "extended"
EVENT_FINISHED = "finished"
EVENT_REOPENED = "reopened"


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def validate_budget(n: Any) -> int:
    """A budget is a positive whole number of METHODS."""
    try:
        v = int(float(n))
    except (TypeError, ValueError):
        raise ValueError(f"The run budget must be a whole number of methods, not {n!r}.")
    if v < 1:
        raise ValueError(
            "The run budget must be at least 1 method. A campaign that "
            "cannot afford a single method is not a campaign.")
    return v


def validate_reason(reason: Any, what: str) -> str:
    """Extensions and endings must say why. This is the whole point of the
    history: an unexplained extension records that the ceiling moved but not
    what moved it, which is the ambiguity the record is meant to remove."""
    r = str(reason or "").strip()
    if not r:
        raise ValueError(
            f"Give a reason for {what}. It is written to the campaign's "
            f"budget history and appears in the exported report, so that "
            f"anyone reading the campaign later can see why it ran as long "
            f"as it did.")
    return r


def event(kind: str, *, used: int, reason: str = "",
          frm: Optional[int] = None, to: Optional[int] = None) -> dict[str, Any]:
    """One line of the campaign's lifecycle history.

    `used` is stamped because WHEN an extension happened is most of its
    meaning: forty methods extended at method 39 is a campaign that ran out of
    road, the same extension made at method 12 is a change of plan.
    """
    ev: dict[str, Any] = {"event": kind, "at": _now(), "used": int(used)}
    if frm is not None:
        ev["from"] = int(frm)
    if to is not None:
        ev["to"] = int(to)
        if frm is not None:
            ev["n"] = int(to) - int(frm)
    if reason:
        ev["reason"] = reason
    return ev


@dataclass
class BudgetStatus:
    """What the budget allows right now, and why."""

    used: int                       # DISTINCT design methods recorded
    limit: int                      # the current ceiling, in methods
    original: int                   # the ceiling the campaign was created with
    n_replicates: int               # injections one method costs; not budgeted
    need: int = 1                   # methods the next thing costs: 1, or 0 for
                                    # an instrument check
    injections: int = 0             # every row recorded; reported, never gated
    finished: bool = False          # the analyst closed it
    end_reason: str = ""
    ended_at: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)

    # ── arithmetic ──────────────────────────────────────────────────────
    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def over(self) -> bool:
        """Recorded more methods than the ceiling. Reachable by lowering a budget
        on a campaign already past it; treated as exhausted, never as an error,
        because the runs happened and the data is real."""
        return self.used > self.limit

    @property
    def fits(self) -> bool:
        """Is there room for the next thing?

        `need` is honoured as given, including 0. The old floor of `max(1,
        need)` was right when everything cost at least one injection; an
        instrument check now costs no budget, and a floor would have made a
        free thing unaffordable.
        """
        return self.remaining >= self.need

    @property
    def methods_left(self) -> int:
        """How many more methods are left. The budget is already in methods, so
        this no longer divides by anything - kept as a name because every
        caller asks for it and the question has not changed."""
        return self.remaining

    @property
    def cost(self) -> dict[str, int]:
        """The whole budget's cost on the instrument, if it is all spent."""
        return injection_cost(self.limit, self.n_replicates)

    @property
    def injections_left(self) -> int:
        """Roughly how many injections the remaining methods will take."""
        return self.remaining * max(1, self.n_replicates)

    @property
    def fraction(self) -> float:
        return (self.used / self.limit) if self.limit > 0 else 1.0

    @property
    def n_extensions(self) -> int:
        return sum(1 for e in self.history if e.get("event") == EVENT_EXTENDED)

    @property
    def extended_by(self) -> int:
        return self.limit - self.original

    # ── the verdict ─────────────────────────────────────────────────────
    @property
    def exhausted(self) -> bool:
        """The plan is spent - a question about the BUDGET, not about what
        happens to be next. It used to be `not fits`, which was the same thing
        while everything cost at least one. Now that an instrument check costs
        nothing, `not fits` would say a full campaign still had room whenever a
        reference was the next thing due."""
        return not self.finished and self.remaining <= 0

    @property
    def warn(self) -> bool:
        """Near the end, but still able to serve the next thing."""
        return (not self.finished and self.fits
                and self.methods_left <= WARN_METHODS)

    @property
    def blocks_proposing(self) -> bool:
        return self.finished or self.exhausted

    @property
    def state(self) -> str:
        if self.finished:
            return "FINISHED"
        if self.exhausted:
            return "EXHAUSTED"
        return "WATCH" if self.warn else "OK"

    def _m(self, n: int) -> str:
        return f"{n} method{'' if n == 1 else 's'}"

    def reason(self) -> str:
        """Every sentence is in METHODS, with the injections named beside them.
        The budget is the plan; the injections are the bench time it costs, and
        an analyst needs both to decide whether to keep going."""
        if self.finished:
            r = f" Reason given: {self.end_reason}" if self.end_reason else ""
            return (f"This campaign was closed by the analyst"
                    f"{' on ' + self.ended_at[:10] if self.ended_at else ''} "
                    f"after {self.used} of {self._m(self.limit)}, "
                    f"{self.injections} injections.{r}")
        if self.over:
            return (f"{self._m(self.used)} are recorded against a budget of "
                    f"{self.limit}. The budget was lowered below the methods "
                    f"already run; nothing is wrong with the data.")
        if self.exhausted:
            return (f"The {self.limit}-method budget is used up: {self.used} "
                    f"of {self.limit} methods recorded, {self.injections} "
                    f"injections spent.")
        if self.warn:
            return (f"{self.used} of {self._m(self.limit)} used. "
                    f"{self._m(self.remaining)} left - about "
                    f"{self.injections_left} more injection"
                    f"{'' if self.injections_left == 1 else 's'} at "
                    f"{self.n_replicates} repeat"
                    f"{'' if self.n_replicates == 1 else 's'} each.")
        return (f"{self.used} of {self._m(self.limit)} used, "
                f"{self._m(self.remaining)} left - about "
                f"{self.injections_left} more injections at "
                f"{self.n_replicates} repeat"
                f"{'' if self.n_replicates == 1 else 's'} each.")

    def actions(self) -> list[str]:
        if self.finished:
            return ["The record stays readable and exportable. Reopen the "
                    "campaign if you need more methods against the same model."]
        if self.exhausted:
            return [
                "Finish the campaign if the separation is good enough - "
                "the best method and its chromatogram are already recorded.",
                "Or extend the budget by however many methods the remaining "
                "question needs, with a reason. The extension is written to "
                "the campaign history and appears in the report.",
                "Before extending, look at whether the campaign is still "
                "learning: a flat uncertainty trend and a reference still "
                "passing means more methods buy precision, not a better one.",
            ]
        if self.warn:
            return [
                f"{self._m(self.remaining)} left, about "
                f"{self.injections_left} injections. Decide now whether they "
                f"go to exploring or to confirming the best method so far.",
                "Instrument checks are free against this budget - they cost "
                "bench time and spend no methods.",
            ]
        return ["Continue the loop."]

    def as_dict(self) -> dict[str, Any]:
        return {
            "used": self.used, "limit": self.limit, "original": self.original,
            "remaining": self.remaining, "need": self.need,
            "n_replicates": self.n_replicates,
            "methods_left": self.methods_left, "fraction": self.fraction,
            "injections": self.injections,
            "injections_left": self.injections_left, "cost": self.cost,
            "state": self.state, "finished": self.finished,
            "exhausted": self.exhausted, "warn": self.warn, "over": self.over,
            "blocks": self.blocks_proposing,
            "end_reason": self.end_reason, "ended_at": self.ended_at,
            "n_extensions": self.n_extensions, "extended_by": self.extended_by,
            "history": list(self.history),
            "reason": self.reason(), "actions": self.actions(),
        }


def plan_extension(status: BudgetStatus, n: Any, reason: Any) -> tuple[int, dict]:
    """Validate an extension and return (new_limit, history event). In METHODS.

    Refuses on a finished campaign: reopen it first, so that the reopening is
    itself recorded rather than being smuggled in as an extension.
    """
    if status.finished:
        raise ValueError(
            "This campaign is closed. Reopen it before extending the budget, "
            "so the record shows it was reopened and why.")
    n = validate_budget(n)
    why = validate_reason(reason, "the extension")
    new_limit = status.limit + n
    return new_limit, event(EVENT_EXTENDED, used=status.used, reason=why,
                            frm=status.limit, to=new_limit)


def plan_finish(status: BudgetStatus, reason: Any) -> dict:
    """Validate closing a campaign and return the history event."""
    if status.finished:
        raise ValueError("This campaign is already closed.")
    return event(EVENT_FINISHED, used=status.used,
                 reason=validate_reason(reason, "finishing the campaign"))


def plan_reopen(status: BudgetStatus, reason: Any) -> dict:
    """Validate reopening a closed campaign and return the history event.

    Reopening is allowed - closing by mistake should not cost a campaign - but
    it is recorded, because a campaign that was declared finished and then
    continued has an ending that no longer means what it said.
    """
    if not status.finished:
        raise ValueError("This campaign is not closed.")
    return event(EVENT_REOPENED, used=status.used,
                 reason=validate_reason(reason, "reopening the campaign"))
