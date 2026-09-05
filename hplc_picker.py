"""hplc_picker - compatibility shim. THE REAL MODULE MOVED INTO THE PACKAGE.

The peak picker now lives at `gradient_bench/core/hplc_picker.py`, which is the
single source of truth for what a peak is.

It used to sit here as a loose module, with a second copy in `v3/` that
`build_nb.py` regenerated the notebook from. Two copies of the file that defines
the objective's inputs is a drift hazard of the worst kind: edit one and the
other goes stale SILENTLY, and the app and the notebook then disagree about how
many peaks a chromatogram has, with nothing anywhere to say so.

This file exists so that `import hplc_picker` keeps working for anything that
expects the old layout - the notebook's bootstrap, an analyst's own script, a
stale sys.path. It is three lines and it cannot drift, because it does not
contain a copy of anything: it hands back the very module object the package
imports.

If you are editing the picker, edit `gradient_bench/core/hplc_picker.py` and
re-run the tests. Then run `python3 v3/build_nb.py` to regenerate the notebook's
embedded copy from it.
"""
import sys

from gradient_bench.core import hplc_picker as _real

sys.modules[__name__] = _real
