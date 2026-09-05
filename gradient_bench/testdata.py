"""Where the bundled corpus lives.

Twenty real chromatograms ship in `testdata/traces/`. They are what the golden
file tests adjudicate against, what `replay_corpus` replays, and what
`verify_install` scores to prove the picker on this machine agrees with the
picker the objective was frozen against.

This is two functions and no policy on purpose. It used to live inside
`demo.py`; when the demonstration campaign was removed these came with it,
because locating the corpus is a fact about where the package keeps its files
and has nothing to do with the feature that happened to be its first caller.
"""
from __future__ import annotations

import os

#: Found relative to the package rather than the working directory, so it
#: resolves from a copied folder - which is how this app is expected to travel.
def traces_dir() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "testdata", "traces")


def available() -> bool:
    """True when enough of the corpus is present to be worth scoring.

    Eight is a floor, not a count: a folder with one or two stray files is a
    broken install rather than a small corpus, and saying so early is more use
    than failing later inside the picker.
    """
    d = traces_dir()
    return os.path.isdir(d) and len(
        [f for f in os.listdir(d) if f.lower().endswith(".txt")]) >= 8
