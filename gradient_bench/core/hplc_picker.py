"""
hplc_picker - HPLC peak picking
================================================================================

One configuration, one pipeline pass, and pure formatters over the result.

  * ONE CONFIG.  Every tunable lives in `PickerConfig`.  Nothing in this module
    reads a global and no function hardcodes a threshold, so changing `snr` in
    one place moves the detection, the categorisation, the hump gate, the
    feature dict AND the printed report together.  There is no second copy of
    any number to fall out of step with the master sheet.

  * ONE PIPELINE PASS.  `analyse()` runs the whole chain exactly once and
    returns a `PickerResult` carrying every intermediate.  `extract_features()`
    and `feature_report()` are pure formatters over that result - they cannot
    recompute, therefore they cannot disagree.

  * WINDOWS ARE MINUTES, NOT SAMPLES.  Chromatograms are exported at whatever
    rate the instrument software was set to, and the same physical peak must
    not be smoothed, or width-gated, differently because of that setting.  So
    every window is specified in MINUTES and converted to a sample count
    against the trace's own median dt by `PickerConfig.n_samples()`.  The
    defaults are quoted at 60 points/minute.

  * RATE-INVARIANT NOISE AND BASELINE.  `estimate_noise` differences the trace
    at a fixed TIME lag (`cfg.noise_lag_min`), and the arPLS smoothness penalty
    `arpls_lam` is quoted at a reference acquisition rate and rescaled to the
    trace's own rate.  Both quantities are then properties of the chemistry and
    the detector rather than of how densely the trace was written to disk,
    which is what makes two runs comparable with each other.

  * TRACE HEALTH.  `analyse()` also returns a `trace_health` block (surfaced by
    `extract_features()` and listed in `TRACE_HEALTH_KEYS`) giving acquisition,
    detector-scale and peak-shape descriptors for drift monitoring.  These are
    DIAGNOSTICS ONLY: `compute_crf` does not read them and `CRF_COLUMNS` does
    not contain them.

THE HUMP THRESHOLDS
--------------------------------------------------------------------------------
`PickerConfig` is the only place the UCM hump thresholds appear
(`hump_min_floor_ratio = 0.95`, `hump_min_span_abs = 1.0` min).  No hump
function carries a threshold in its signature, so there is nothing for the
config to disagree with.  The hump call - and through it the clean-peak count,
which is the objective - is sensitive to those two numbers, so
`PickerConfig.with_relaxed_hump_thresholds()` returns a deliberately looser
pair and the sensitivity can be measured rather than assumed.

Usage
--------------------------------------------------------------------------------
    from hplc_picker import PickerConfig, analyse, extract_features, feature_report

    cfg = PickerConfig(snr=5.0)
    res = analyse("method19.TXT", cfg)
    ft  = extract_features(res)          # dict for the master sheet
    print(feature_report(res, cfg))      # same numbers, formatted
    res.trace_health                     # drift-monitor descriptors
"""

from __future__ import annotations

import os
import glob
import datetime as _datetime
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import savgol_filter, find_peaks, peak_widths, argrelmin
from scipy.special import erfc, erfcx
from scipy.optimize import curve_fit
from scipy.stats import f as f_dist
from scipy.ndimage import minimum_filter1d, maximum_filter1d
from pybaselines import Baseline

__version__ = "3.0.0"

__all__ = [
    "__version__",
    "PickerConfig", "DEFAULT_CONFIG", "PickerResult",
    "analyse", "extract_features", "feature_report",
    "CRF_COLUMNS", "compute_crf",
    "MEASUREMENT_KEYS", "measurements_dict",
    "RECORD_KEYS", "RECORD_FEATURE_KEYS", "record_dict",
    "FEATURE_KEYS", "TRACE_HEALTH_KEYS",
    "load_chromatogram", "estimate_noise", "list_files",
    "emg", "multi_emg", "usp_tailing",
]


# ══════════════════════════════════════════════════════════════════════════════
#  1 · CONFIGURATION - the single source of truth
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PickerConfig:
    """Every tunable in the pipeline, in one place.

    All *_min fields are in MINUTES and are converted to an odd sample count
    against the trace's own median dt (see `n_samples`).  The defaults are
    quoted at 60 points/minute; at any other rate the same physical window is
    used, which is the whole point of specifying them in time.
    """

    # ---- baseline ------------------------------------------------------------
    baseline_method: str = "arpls"        # 'arpls' | 'snip'
    arpls_lam: float = 1e5                # arPLS smoothness penalty AT the rate
                                          # named by arpls_lam_ref_pts_per_min
    # arPLS penalises the SECOND DIFFERENCE between neighbouring SAMPLES, so a
    # fixed lam is a different physical stiffness at every acquisition rate:
    # the penalty scales as dt**4 against a fit term that scales as dt**-1, i.e.
    # rate-invariance needs lam proportional to pts_per_min**4.  Set this to
    # None to switch the rescaling off and use arpls_lam literally.
    arpls_lam_ref_pts_per_min: Optional[float] = 60.0
    snip_max_half_window: int = 40        # SNIP window (samples) if method='snip'
    snip_smooth_half_window: int = 3
    clip_negative: bool = True            # clamp corrected/smoothed trace at 0

    # ---- smoothing (Savitzky-Golay) -----------------------------------------
    sg_window_min: float = 11.0 / 60.0    # MINUTES  (= 11 samples @ 60 pts/min)
    sg_poly: int = 3

    # ---- prominence detection ------------------------------------------------
    snr: float = 5.0                      # prominence gate = snr * eff_noise
    min_width_min: float = 3.0 / 60.0     # MINUTES  (= 3 samples @ 60 pts/min)
    noise_floor_frac: float = 0.0015      # eff_noise = max(noise, frac * ymax)

    # ---- second-derivative (shoulder) detection ------------------------------
    d2_snr: float = 3.0                   # -d2 trough prominence / sigma_d2
    d2_window_min: float = 21.0 / 60.0    # MINUTES  (= 21 samples @ 60 pts/min)
    d2_poly: int = 3
    d2_concavity_margin: float = 1.5      # apex must have d2 < -margin*sigma_d2

    # ---- hybrid fit-gated shoulder acceptance --------------------------------
    # Detection mode: 'prominence' | 'second_derivative' | 'hybrid' |
    # 'hybrid_fit'.  analyse() always keeps BOTH the plain prominence pass and
    # the pass named here, because the categorisation needs both: a pick counts
    # as CLEAN only if the prominence pass found it too.
    detection: str = "hybrid_fit"
    hybrid_p_thresh: float = 0.01         # partial F-test significance
    hybrid_min_rel: float = 0.003         # min relative SSE reduction
    hybrid_shoulder_snr: float = 5.0      # new EMG component height / eff_noise
    hybrid_max_shoulders: int = 20        # per cluster, ranked by height
    hybrid_min_sep_fwhm_frac: float = 0.6  # local resolution guard
    merge_min_sep_min: float = 4.0 / 60.0  # MINUTES (= 4 samples @ 60 pts/min)

    # ---- EMG deconvolution ---------------------------------------------------
    deconvolve: bool = True
    max_cluster: int = 15                 # was module-level MAX_CLUSTER
    cluster_gap_min: float = 2.0 / 60.0   # MINUTES (= 2 samples @ 60 pts/min)
    fit_pad_min: float = 5.0 / 60.0       # MINUTES (= 5 samples @ 60 pts/min)
    fit_maxfev: int = 4000
    fit_min_r2: float = 0.5               # below this, fall back to raw comps
    emg_area_factor: float = 1.064        # h*FWHM -> area seed
    emg_sigma_lo_frac: float = 0.2        # lb sigma = frac * dt
    emg_tau_lo_frac: float = 0.05         # lb tau   = frac * dt
    emg_sigma_hi_frac: float = 4.0        # ub sigma = frac * FWHM
    emg_tau_hi_frac: float = 6.0          # ub tau   = frac * FWHM
    emg_amp_hi_factor: float = 50.0

    # ---- resolution scorecard ------------------------------------------------
    rs_baseline: float = 1.5              # textbook baseline-separation Rs
    valley_frac: float = 0.10             # valley/min-height "returned to base"
    rs_clip: float = 2.5                  # cap on Rs before summing in features

    # ---- UCM hump detection --------------------------------------------------
    # THE ONLY hump thresholds anywhere in this module.  No hump function
    # carries a threshold in its signature, so there is nothing for these to
    # disagree with.  They are also the numbers the hump call is most sensitive
    # to; `with_relaxed_hump_thresholds()` returns a looser pair so that the
    # sensitivity can be measured rather than assumed.
    hump_snr: Optional[float] = None      # None -> follow cfg.snr
    hump_min_real_peaks: int = 1
    hump_min_span_factor: float = 4.0
    hump_min_span_abs: float = 1.0        # MINUTES
    hump_level_frac: float = 0.005
    hump_min_floor_ratio: float = 0.95
    hump_core_noise_mult: float = 8.0
    hump_extend_noise_mult: float = 4.0
    hump_extend_frac: float = 0.4
    hump_wiggle_prom_frac: float = 0.02
    hump_fallback_fwhm_min: float = 0.2   # MINUTES, when < 3 peaks found
    hump_merge_gap: float = 4.5           # MINUTES (0 = off)
    hump_grow_clean_frac: float = 0.03
    hump_grow_bump_hmax: float = 0.10
    hump_split: bool = False
    hump_split_dip_frac: float = 0.35
    hump_split_min_sub_span: float = 1.5  # MINUTES
    hump_override: Optional[Sequence[Tuple[float, float]]] = None
    # --- hump windows: MINUTES, so the hump detector scales with the
    #     acquisition rate exactly the way the smoothing windows do.  A floor
    #     expressed in SAMPLES here would be a long physical window on a slow
    #     export and a short one on a fast export.
    hump_min_fwhm_min: float = 3.0 / 60.0     # floor on the median peak width
                                              # used to size the hump windows
                                              # (= 3 samples @ 60 pts/min)
    hump_envelope_span_factor: float = 2.0    # lower-envelope window
                                              # = factor * median peak width
    hump_envelope_min_min: float = 7.0 / 60.0  # floor on it (= 7 samples @60)
    hump_fallback_floor_min: float = 5.0 / 60.0  # floor when < 3 peaks found
    hump_split_min_samples: int = 6           # a region shorter than this many
                                              # samples is never split

    # ---- misc ----------------------------------------------------------------
    min_window_samples: int = 3           # absolute floor for n_samples()
    min_width_floor_samples: float = 1.0  # absolute floor for the width gate

    # ---- noise (rate-invariant) ----------------------------------------------
    # sigma is the robust SD of the trace differenced at a fixed TIME lag.  See
    # `estimate_noise` for the derivation, and for why a fixed SAMPLE lag is
    # not comparable between exports.
    noise_lag_min: float = 3.0 / 60.0     # MINUTES (= 3 samples @ 60 pts/min)
    noise_override: Optional[float] = None  # skip the estimate, use this sigma

    # ------------------------------------------------------------------ helper
    def n_samples(self, minutes: float, dt: float, *,
                  floor: Optional[int] = None,
                  poly: Optional[int] = None,
                  n_total: Optional[int] = None,
                  odd: bool = True) -> int:
        """Convert a window in MINUTES to a sample count for this trace.

        This is the one place samples are ever derived from time, so every
        window in the pipeline scales with the acquisition rate identically.

        floor    : minimum sample count (default cfg.min_window_samples)
        poly     : if given, also enforce w > poly the way savgol needs
        n_total  : trace length; the window is shrunk to fit inside it
        odd      : round to the NEAREST ODD sample count (savgol requires odd;
                   separations/paddings pass odd=False)
        """
        if floor is None:
            floor = self.min_window_samples
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0
        raw = float(minutes) / float(dt)
        if not np.isfinite(raw) or raw <= 0:
            raw = float(floor)
        if odd:
            w = int(round((raw - 1.0) / 2.0)) * 2 + 1     # nearest odd integer
        else:
            w = int(round(raw))
        w = max(w, int(floor))
        if poly is not None:
            # savgol needs window > poly
            w = max(w, poly + 2 + ((poly + 1) % 2))
            if odd and w % 2 == 0:
                w += 1
        if n_total is not None and w >= n_total:
            if odd:
                w = n_total - 1 if (n_total - 1) % 2 == 1 else n_total - 2
            else:
                w = max(n_total - 1, 1)
        return int(max(w, 1))

    def width_samples(self, dt: float) -> float:
        """The minimum-width GATE in samples.

        Deliberately NOT routed through `n_samples`: a width gate is not a
        filter window, so it must not be forced odd and must not carry a
        multi-sample floor.  Such a floor is itself a sample-based rule, and on
        a low-rate export it silently inflates the gate (0.05 min asked for,
        0.176 min applied at 17 pts/min).  `find_peaks` accepts a float width,
        so the gate stays honestly time-based; the only floor is 1 sample,
        below which the gate is vacuous anyway.

        At 60 pts/min the default 3/60 min gives exactly 3.0 samples.
        """
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0
        return max(float(self.min_width_min) / float(dt),
                   float(self.min_width_floor_samples))

    # convenience: the snr the hump detector should use
    @property
    def effective_hump_snr(self) -> float:
        return self.snr if self.hump_snr is None else self.hump_snr

    def effective_arpls_lam(self, dt: float) -> float:
        """arPLS lam corrected to this trace's acquisition rate.

        `arpls_lam` is quoted at `arpls_lam_ref_pts_per_min` points/minute.
        The Whittaker penalty is lam * sum_i (second difference between
        SAMPLES)^2; writing the second difference as y''(t) * dt^2 and the
        sums as integrals / dt gives

            penalty / fit  ~  lam * dt^4 * INT(y'')^2 / INT(y - z)^2

        so holding the physical stiffness fixed requires lam ~ dt^-4, i.e.
        lam ~ pts_per_min^4.  Without this a decimated trace gets a far
        stiffer baseline, its corrected signal over a broad region sits higher,
        and the hump detector's `valley_floor_ratio` gate can flip on nothing
        but the export rate.
        """
        ref = self.arpls_lam_ref_pts_per_min
        if ref is None or not np.isfinite(dt) or dt <= 0 or ref <= 0:
            return float(self.arpls_lam)
        return float(self.arpls_lam) * ((1.0 / float(dt)) / float(ref)) ** 4

    def with_relaxed_hump_thresholds(self) -> "PickerConfig":
        """A deliberately looser hump gate: floor ratio 0.85, min span 2.0 min.

        For sensitivity testing.  The hump call - and through it the clean-peak
        count, which is the objective - depends strongly on these two numbers,
        so it is worth being able to re-run a trace with a different pair and
        see how far the answer moves.
        """
        return replace(self, hump_min_floor_ratio=0.85, hump_min_span_abs=2.0)

    def replace(self, **kw) -> "PickerConfig":
        """Return a copy with fields overridden (config stays immutable)."""
        return replace(self, **kw)

    def summary_lines(self) -> List[str]:
        """Plain-ASCII echo of the settings that drive the numbers."""
        return [
            "  baseline           %-12s lam=%-9g @%s pts/min  clip_negative=%s"
            % (self.baseline_method, self.arpls_lam,
               ("%g" % self.arpls_lam_ref_pts_per_min)
               if self.arpls_lam_ref_pts_per_min else "any (no rescale)",
               self.clip_negative),
            "  noise              rate-invariant, lag=%.4f min%s"
            % (self.noise_lag_min,
               "" if self.noise_override is None
               else "   OVERRIDDEN with %.6g" % self.noise_override),
            "  smoothing          sg_window=%.4f min  poly=%d"
            % (self.sg_window_min, self.sg_poly),
            "  detection          mode=%-12s snr=%.2f  min_width=%.4f min"
            "  noise_floor=%.4g"
            % (self.detection, self.snr, self.min_width_min,
               self.noise_floor_frac),
            "  2nd derivative     d2_snr=%.2f  d2_window=%.4f min  poly=%d"
            % (self.d2_snr, self.d2_window_min, self.d2_poly),
            "  hybrid fit gate    p<%.3g  min_rel=%.4g  shoulder_snr=%.2f"
            % (self.hybrid_p_thresh, self.hybrid_min_rel, self.hybrid_shoulder_snr),
            "  humps              snr=%.2f  floor_ratio=%.2f  min_span=%.2f min"
            "  merge_gap=%.2f min"
            % (self.effective_hump_snr, self.hump_min_floor_ratio,
               self.hump_min_span_abs, self.hump_merge_gap),
            "  resolution         Rs_baseline=%.2f  valley_frac=%.2f  Rs_clip=%.2f"
            % (self.rs_baseline, self.valley_frac, self.rs_clip),
        ]


DEFAULT_CONFIG = PickerConfig()


# ══════════════════════════════════════════════════════════════════════════════
#  2 · LOADING & NOISE
# ══════════════════════════════════════════════════════════════════════════════

def load_chromatogram(path):
    """Return (t, y) float arrays from a 2-column scientific-notation file."""
    data = np.loadtxt(path)
    t = data[:, 0].astype(float)
    y = data[:, 1].astype(float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    order = np.argsort(t)
    t, y = t[order], y[order]
    return t, y


def _mad(a):
    return float(np.median(np.abs(a - np.median(a))))


def estimate_noise(y, dt=None, cfg: "PickerConfig" = None):
    """Rate-invariant robust noise sigma.  Returns sigma in y units.

    WHY THE LAG IS A TIME AND NOT A SAMPLE COUNT
    ---------------------------------------------------------------------
    The obvious estimator, `1.4826 * MAD(diff(y)) / sqrt(2)`, takes the MAD of
    the FIRST difference.  Its `/sqrt(2)` is only correct if consecutive
    samples are independent.  A chromatogram exported at 60 points/minute is
    heavily oversampled relative to the detector's own time constant, so
    consecutive samples are strongly correlated, their difference is small, and
    the estimate collapses.  Decimate the SAME trace and the correlation
    breaks, so the "noise" appears to grow: measured over 17 traces resampled
    to 60 / 30 / 17.05 pts/min, a fixed-SAMPLE-lag estimate spans a factor of
    2.36 on average (worst trace 2.90).  A quantity that moves by 2.4x when
    nothing but the export setting changed cannot be used to compare runs,
    which is precisely what a drift monitor has to do.

    THE ESTIMATOR
    ---------------------------------------------------------------------
    Difference at a fixed TIME lag L = `cfg.noise_lag_min` instead of at a
    fixed sample lag:

        d(t) = y(t + L) - y(t)      Var d = 2 (gamma(0) - gamma(L))

    gamma is the autocovariance of the noise process, so Var d is a property
    of the CHEMISTRY AND THE DETECTOR at the timescale L, not of how densely
    the trace happened to be written to disk.  y(t + L) is taken by linear
    interpolation between the two bracketing samples, so L is honoured exactly
    rather than rounded to whole samples (integer rounding alone left a 1.28x
    residual spread).  The variance the interpolation adds is divided out:
    for a blend (1-a) y[k] + a y[k+1] against y[0] the white-noise variance of
    the difference is 1 + (1-a)^2 + a^2 in units of sigma^2.

    MAD (not SD) makes it blind to the peaks: as long as peaks occupy less
    than half the trace, the median absolute deviation of d is set by the
    peak-free stretches.

    L defaults to 3/60 min - the same 3 samples-at-60-pts/min scale as the
    minimum peak-width gate.  That is the right timescale on purpose: sigma is
    used as `snr * sigma` against a peak PROMINENCE, so the fluctuation that
    matters is the one over the shortest interval a peak is allowed to
    occupy, not the sample-to-sample jitter of an oversampled trace.

    MEASURED INVARIANCE (17 traces, resampled to 60 / 30 / 17.05 pts/min):

        estimator                        mean max:min   median   worst
        fixed SAMPLE lag (1st difference)       2.36     2.28     2.90
        fixed TIME lag of 0.05 min              1.06     1.06     1.14

    `cfg.noise_override` is the only way to supply a sigma from outside.

    Parameters
    ----------
    y   : baseline-corrected signal, BEFORE negative clipping.
    dt  : median sample spacing in minutes.  If None the lag degenerates to a
          single sample and the estimate is NOT rate-invariant - callers
          inside this module always pass it.
    cfg : supplies `noise_lag_min`; defaults to DEFAULT_CONFIG.
    """
    cfg = DEFAULT_CONFIG if cfg is None else cfg
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return 1e-9
    if dt is None or not np.isfinite(dt) or dt <= 0:
        k, a = 1, 0.0
    else:
        ratio = float(cfg.noise_lag_min) / float(dt)
        if not np.isfinite(ratio) or ratio < 1.0:
            k, a = 1, 0.0
        else:
            k = int(np.floor(ratio))
            a = ratio - k
    if k + 1 >= n:                      # lag longer than the trace
        k, a = 1, 0.0
    if k + 1 < n:
        shifted = (1.0 - a) * y[k:n - 1] + a * y[k + 1:n]
        d = shifted - y[0:n - 1 - k]
        gain = np.sqrt(1.0 + ((1.0 - a) ** 2 + a ** 2))
    else:
        d = np.diff(y)
        gain = np.sqrt(2.0)
    if len(d) == 0:
        return 1e-9
    return max(1.4826 * _mad(d) / gain, 1e-9)


def list_files(data_dir):
    fs = glob.glob(os.path.join(data_dir, "method*.TXT"))
    fs.sort(key=lambda p: int("".join(ch for ch in os.path.basename(p)
                                      if ch.isdigit())))
    return fs


# ══════════════════════════════════════════════════════════════════════════════
#  3 · BASELINE (arPLS) & SMOOTHING (Savitzky-Golay)
# ══════════════════════════════════════════════════════════════════════════════

def _correct_baseline(t, y, cfg: PickerConfig, dt: Optional[float] = None):
    """Return (corrected_y, baseline).

    `lam` is rate-corrected by `cfg.effective_arpls_lam(dt)` so the baseline
    has the same physical stiffness at any acquisition rate.
    """
    if dt is None:
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    fitter = Baseline(x_data=t)
    if cfg.baseline_method == "arpls":
        bkg, _ = fitter.arpls(y, lam=cfg.effective_arpls_lam(dt))
    elif cfg.baseline_method == "snip":
        bkg, _ = fitter.snip(y, max_half_window=cfg.snip_max_half_window,
                             decreasing=True,
                             smooth_half_window=cfg.snip_smooth_half_window)
    else:
        raise ValueError(cfg.baseline_method)
    return y - bkg, bkg


def _smooth(y, dt, cfg: PickerConfig):
    """Savitzky-Golay smoothing with a TIME-based window."""
    w = cfg.n_samples(cfg.sg_window_min, dt, poly=cfg.sg_poly, n_total=len(y))
    if w <= cfg.sg_poly:
        return y.copy()
    return savgol_filter(y, w, cfg.sg_poly)


def _sg_secondderiv(y, dt, cfg: PickerConfig):
    """Savitzky-Golay second derivative d2y/dt2 with a TIME-based window."""
    w = cfg.n_samples(cfg.d2_window_min, dt, poly=cfg.d2_poly, n_total=len(y))
    if w <= cfg.d2_poly:
        return np.zeros_like(y)
    return savgol_filter(y, w, cfg.d2_poly, deriv=2, delta=dt)


def _empty_detection():
    return dict(idx=np.array([], dtype=int), t=np.array([]),
                height=np.array([]), prom=np.array([]), fwhm_t=np.array([]),
                left_ips=np.array([]), right_ips=np.array([]))


# ══════════════════════════════════════════════════════════════════════════════
#  4 · PROMINENCE DETECTION (the workhorse)
# ══════════════════════════════════════════════════════════════════════════════

def _detect_peaks(t, y, noise, dt, cfg: PickerConfig):
    """Detect peaks on a baseline-corrected, smoothed trace.

        effective_noise = max(measured noise, noise_floor_frac * max(signal))
        prominence gate = snr * effective_noise
        width gate      = cfg.min_width_min converted to samples

    The noise floor (fraction of the tallest peak, a proxy for the detector
    scale) only bites when the measured MAD-noise collapses to ~0 on
    simulated / zero-baseline traces; on real noisy traces the measured noise
    dominates.
    """
    ymax = np.nanmax(y)
    eff_noise = max(noise, cfg.noise_floor_frac * ymax)
    prom = cfg.snr * eff_noise
    min_width = cfg.width_samples(dt)
    idx, props = find_peaks(y, prominence=prom, width=min_width)
    if len(idx) == 0:
        return _empty_detection()
    w_res = peak_widths(y, idx, rel_height=0.5)
    fwhm_t = w_res[0] * dt
    return dict(idx=idx, t=t[idx], height=y[idx],
                prom=props["prominences"], fwhm_t=fwhm_t,
                left_ips=w_res[2], right_ips=w_res[3])


# ══════════════════════════════════════════════════════════════════════════════
#  5 · SECOND-DERIVATIVE SHOULDER DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _half_height_footprint(y, idx):
    """Half-height footprint of points that need NOT be local maxima.

    `scipy.signal.peak_widths` is only defined for local maxima: it derives
    the reference level from `peak_prominences`, which walks outward while the
    trace stays BELOW the peak value.  A second-derivative shoulder sits on the
    flank of a bigger peak, so one side never comes back down; scipy emits a
    `PeakPropertyWarning` and returns a prominence of 0, hence a width of 0.
    A zero-width footprint would then make `_cluster_peaks` see a point-like
    peak and split clusters that should have been joined.

    Here the footprint is defined directly and unconditionally: walk left and
    right from `idx` until the signal first falls below half the value AT
    `idx`, and linearly interpolate the crossing.  If a side never falls below
    the half level (a shoulder buried in a bigger peak) the trace end is used.
    The footprint is always at least +/- 1 sample wide, so it can never be
    degenerate.

    Returns (widths_samples, left_ips, right_ips) as float arrays, matching
    elements 0, 2 and 3 of `scipy.signal.peak_widths`.
    """
    y = np.asarray(y, dtype=float)
    idx = np.asarray(idx, dtype=int)
    n = len(y)
    L = np.empty(len(idx), dtype=float)
    R = np.empty(len(idx), dtype=float)
    for m, i in enumerate(idx):
        half = 0.5 * y[i]
        j = i
        while j > 0 and y[j] > half:
            j -= 1
        if y[j] > half:                      # never crossed: clamp to the end
            left = float(j)
        elif j == i:
            left = float(i)
        else:
            span = y[j + 1] - y[j]
            frac = (half - y[j]) / span if span > 0 else 0.0
            left = j + float(np.clip(frac, 0.0, 1.0))
        j = i
        while j < n - 1 and y[j] > half:
            j += 1
        if y[j] > half:
            right = float(j)
        elif j == i:
            right = float(i)
        else:
            span = y[j - 1] - y[j]
            frac = (half - y[j]) / span if span > 0 else 0.0
            right = j - float(np.clip(frac, 0.0, 1.0))
        left = min(left, float(i))
        right = max(right, float(i))
        if right - left < 1.0:               # never degenerate
            left = max(left - 0.5, 0.0)
            right = min(right + 0.5, float(n - 1))
        L[m], R[m] = left, right
    return R - L, L, R


def _detect_peaks_2d(t, y, noise, dt, cfg: PickerConfig):
    """Second-derivative peak detection.

    A peak (including an unresolved shoulder) is a local MINIMUM of d2y/dt2.
    We find peaks in -d2y, gated three ways:
      1. depth   : -d2 trough prominence > d2_snr * sigma_d2 (MAD of d2);
      2. concavity: a genuine apex is concave-DOWN, d2 < -margin*sigma_d2, so
         ripples on a smooth convex tail are rejected;
      3. amplitude: the underlying smoothed signal must exceed snr * eff_noise.

    Returns the same dict shape as `_detect_peaks` so it is a drop-in for
    deconvolution and plotting.
    """
    d2 = _sg_secondderiv(y, dt, cfg)
    neg = -d2
    sigma_d2 = 1.4826 * np.median(np.abs(d2 - np.median(d2)))
    sigma_d2 = max(sigma_d2, 1e-12)
    min_width = cfg.width_samples(dt)
    cand, _ = find_peaks(neg, prominence=cfg.d2_snr * sigma_d2, width=min_width)
    cand = cand[d2[cand] < -cfg.d2_concavity_margin * sigma_d2]
    ymax = np.nanmax(y)
    eff_noise = max(noise, cfg.noise_floor_frac * ymax)
    cand = cand[y[cand] > cfg.snr * eff_noise]
    if len(cand) == 0:
        return _empty_detection()
    # NOT peak_widths(): these candidates are minima of d2y, not local maxima
    # of y, so scipy's prominence-based width is undefined for them.  See
    # `_half_height_footprint`.
    w_samp, L, R = _half_height_footprint(y, cand)
    fwhm_t = w_samp * dt
    return dict(idx=cand, t=t[cand], height=y[cand],
                prom=np.full(len(cand), np.nan), fwhm_t=fwhm_t,
                left_ips=L, right_ips=R)


def _merge_detections(t, primary, extra, dt, cfg: PickerConfig):
    """Add 'extra' peaks to 'primary' only where they are not already within
    cfg.merge_min_sep_min of an existing primary peak."""
    min_sep = cfg.n_samples(cfg.merge_min_sep_min, dt, floor=1, odd=False)
    if len(primary["idx"]) == 0:
        return extra
    if len(extra["idx"]) == 0:
        return primary
    keep = [j for j, ix in enumerate(extra["idx"])
            if np.min(np.abs(primary["idx"] - ix)) > min_sep]
    if not keep:
        return primary
    keep = np.array(keep)
    idx = np.concatenate([primary["idx"], extra["idx"][keep]])
    order = np.argsort(idx)
    out = {"idx": idx[order]}
    for k in ("t", "height", "fwhm_t", "left_ips", "right_ips"):
        out[k] = np.concatenate([primary[k], extra[k][keep]])[order]
    out["prom"] = np.concatenate(
        [primary.get("prom", np.full(len(primary["idx"]), np.nan)),
         np.full(len(keep), np.nan)])[order]
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  6 · EMG DECONVOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def emg(x, A, mu, sigma, tau):
    """Exponentially-Modified Gaussian, total area = A, tau>0 -> right tail.

    Numerically stable for all x: uses the scaled complementary error function
    (erfcx) on the rising side where the naive exp*erfc form overflows, and the
    direct form on the falling side."""
    x = np.asarray(x, dtype=float)
    sigma = abs(sigma) + 1e-12
    tau = abs(tau) + 1e-12
    lam = 1.0 / tau
    z = (sigma * lam - (x - mu) / sigma) / np.sqrt(2.0)
    expo = 0.5 * (sigma * lam) ** 2 - lam * (x - mu)
    out = np.zeros_like(x)
    hi = z >= 0                      # rising side / near peak
    out[hi] = (A * lam / 2.0) * np.exp(expo[hi] - z[hi] ** 2) * erfcx(z[hi])
    lo = ~hi                         # falling tail (expo bounded, erfc<=2)
    out[lo] = (A * lam / 2.0) * np.exp(np.clip(expo[lo], -700, 700)) * erfc(z[lo])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def multi_emg(x, *p):
    n = len(p) // 4
    out = np.zeros_like(x, dtype=float)
    for i in range(n):
        A, mu, sigma, tau = p[4 * i:4 * i + 4]
        out += emg(x, A, mu, sigma, tau)
    return out


def _cluster_peaks(idx, left_ips, right_ips, dt, cfg: PickerConfig):
    """Group peaks whose half-max footprints overlap into fit clusters.
    Returns list of lists of positions into the original idx array."""
    if len(idx) == 0:
        return []
    gap = cfg.n_samples(cfg.cluster_gap_min, dt, floor=0, odd=False)
    order = np.argsort(idx)
    L = np.asarray(left_ips)[order]
    R = np.asarray(right_ips)[order]
    clusters, cur = [], [0]
    for k in range(1, len(order)):
        if L[k] <= R[cur[-1]] + gap:
            cur.append(k)
        else:
            clusters.append(cur)
            cur = [k]
    clusters.append(cur)
    return [[order[j] for j in c] for c in clusters]


def _seed_params(t, idx_j, det_like, j, dt, cfg: PickerConfig,
                 xlo: Optional[float] = None, xhi: Optional[float] = None):
    """Build (p0, lb, ub) for one EMG component seeded from a detection.

    `xlo`/`xhi` are the ends of the fit window and they are honoured: the mean
    is bounded inside [xlo, xhi], and the seed is clipped into the same
    interval because `curve_fit` rejects a p0 outside its bounds.  Without that
    bound `curve_fit` is free to walk a component's mean outside the data it is
    being fitted to.
    """
    fw = max(det_like["fwhm_t"][j], 2 * dt)
    A0 = max(det_like["height"][j], 1e-6) * fw * cfg.emg_area_factor
    sig0 = max(fw / 2.355, dt)
    rt = float(t[idx_j])
    mu_lo, mu_hi = rt - fw, rt + fw
    if xlo is not None and xhi is not None and xhi > xlo:
        mu_lo = max(mu_lo, float(xlo))
        mu_hi = min(mu_hi, float(xhi))
        if mu_hi <= mu_lo:                      # degenerate window: reopen it
            mu_lo, mu_hi = float(xlo), float(xhi)
        rt = float(np.clip(rt, mu_lo, mu_hi))
    p0 = [A0, rt, sig0, sig0]
    lb = [0, mu_lo, dt * cfg.emg_sigma_lo_frac, dt * cfg.emg_tau_lo_frac]
    ub = [A0 * cfg.emg_amp_hi_factor + 1e3, mu_hi,
          cfg.emg_sigma_hi_frac * fw, cfg.emg_tau_hi_frac * fw]
    return p0, lb, ub


def _fit_window_sse(xs, ys, seeds, cfg: PickerConfig):
    """Fit a fixed set of EMG seeds on a fixed window; return (sse, popt)."""
    if not seeds:
        return float("inf"), None
    p0 = [v for s in seeds for v in s[0]]
    lb = [v for s in seeds for v in s[1]]
    ub = [v for s in seeds for v in s[2]]
    try:
        popt, _ = curve_fit(multi_emg, xs, ys, p0=p0, bounds=(lb, ub),
                            maxfev=cfg.fit_maxfev)
    except Exception:
        return float("inf"), None
    sse = float(np.sum((ys - multi_emg(xs, *popt)) ** 2))
    return sse, popt


def _raw_component(t, idx_j, det, j, cfg: PickerConfig):
    """Fallback component from detection only (no fit)."""
    return dict(rt=float(t[idx_j]),
                area=float(det["height"][j] * det["fwhm_t"][j]
                           * cfg.emg_area_factor),
                height=float(det["height"][j]),
                sigma=float(det["fwhm_t"][j] / 2.355), tau=float("nan"),
                r2=float("nan"), fitted=False)


def _fit_group(t, y, det, ci, dt, cfg: PickerConfig):
    """Fit a small group of EMG peaks (indices ci into det arrays)."""
    idx = det["idx"]
    L, R = det["left_ips"], det["right_ips"]
    pad = cfg.n_samples(cfg.fit_pad_min, dt, floor=1, odd=False)
    lo = int(max(0, np.floor(L[ci].min()) - pad))
    hi = int(min(len(t) - 1, np.ceil(R[ci].max()) + pad))
    xs, ys = t[lo:hi + 1], y[lo:hi + 1]
    if len(xs) < 5:
        return None, None
    p0, lb, ub = [], [], []
    for j in ci:
        s_p0, s_lb, s_ub = _seed_params(t, idx[j], det, j, dt, cfg,
                                        xlo=xs[0], xhi=xs[-1])
        p0 += s_p0
        lb += s_lb
        ub += s_ub
    try:
        popt, _ = curve_fit(multi_emg, xs, ys, p0=p0, bounds=(lb, ub),
                            maxfev=cfg.fit_maxfev)
    except Exception:
        return None, None
    fit = multi_emg(xs, *popt)
    ss_res = np.sum((ys - fit) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2) + 1e-12
    r2 = float(1 - ss_res / ss_tot)
    comps = []
    for i in range(len(ci)):
        A, mu, sigma, tau = popt[4 * i:4 * i + 4]
        comp = emg(xs, A, mu, sigma, tau)
        comps.append(dict(rt=float(mu), area=float(np.trapezoid(comp, xs)),
                          height=float(comp.max()), sigma=float(abs(sigma)),
                          tau=float(abs(tau)), r2=r2, fitted=True))
    return comps, (lo, hi, fit)


def _deconvolve_emg(t, y, det, dt, cfg: PickerConfig):
    """Fit EMG components cluster-by-cluster.  Large clusters are split into
    chunks of <= cfg.max_cluster neighbouring peaks to keep fits fast and
    stable.  Returns (components, model).

    MODEL OVERLAY.  Chunks of one big cluster are fitted on PADDED windows, so
    consecutive chunk windows overlap by up to 2 * cfg.fit_pad_min.  Pasting
    each chunk's prediction into a shared array would therefore add two
    predictions inside every overlap and inflate the curve there.  The model is
    instead summed from the FITTED COMPONENTS over the full time axis, so every
    component contributes exactly once by construction.  `model` is used for
    plotting only - no feature and no CRF input reads it.
    """
    idx = det["idx"]
    if len(idx) == 0:
        return [], np.zeros_like(y)
    clusters = _cluster_peaks(idx, det["left_ips"], det["right_ips"], dt, cfg)
    components = []
    for cl in clusters:
        ci_all = np.array(sorted(cl))
        for s in range(0, len(ci_all), cfg.max_cluster):
            ci = ci_all[s:s + cfg.max_cluster]
            comps, _fitinfo = _fit_group(t, y, det, ci, dt, cfg)
            # reject poor fits (e.g. overloaded shark-fin fronts EMG cannot
            # represent): keep the peak count via raw fallback, but do not let
            # a bad fit pollute the model overlay or the reported R^2.
            good = comps is not None and comps[0]["r2"] >= cfg.fit_min_r2
            if not good:
                for j in ci:
                    components.append(_raw_component(t, idx[j], det, j, cfg))
            else:
                components.extend(comps)
    components.sort(key=lambda d: d["rt"])
    return components, _reconstruct_model(t, components)


def _reconstruct_model(t, components):
    """Smooth full-axis EMG sum from fitted component parameters (plotting)."""
    model = np.zeros_like(t, dtype=float)
    for c in components:
        if c.get("fitted") and np.isfinite(c.get("tau", np.nan)):
            model += emg(t, c["area"], c["rt"], c["sigma"], c["tau"])
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  7 · HYBRID FIT-GATED DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _detect_hybrid_fitgated(t, y, noise, dt, cfg: PickerConfig,
                            det_p=None, det_d=None):
    """Hybrid detection with an EMG fit-significance gate.

    1. Prominence peaks are always accepted (clear local maxima).
    2. Second-derivative minima that are NOT already a prominence peak become
       *candidate* shoulders.
    3. Per overlapping cluster, candidate shoulders are added by forward
       selection; a shoulder is kept only if including it gives a statistically
       significant reduction in the fit residual (partial F-test,
       p < cfg.hybrid_p_thresh) over the same window, AND a real relative
       residual drop, AND the new component itself rises above noise.
    """
    if det_p is None:
        det_p = _detect_peaks(t, y, noise, dt, cfg)
    if det_d is None:
        det_d = _detect_peaks_2d(t, y, noise, dt, cfg)
    min_sep = cfg.n_samples(cfg.merge_min_sep_min, dt, floor=1, odd=False)

    if len(det_p["idx"]) and len(det_d["idx"]):
        cand = np.array([k for k, ix in enumerate(det_d["idx"])
                         if np.min(np.abs(det_p["idx"] - ix)) > min_sep])
    else:
        cand = np.arange(len(det_d["idx"]))
    if len(cand) == 0:
        return det_p

    all_idx = np.concatenate([det_p["idx"], det_d["idx"][cand]])
    all_L = np.concatenate([det_p["left_ips"], det_d["left_ips"][cand]])
    all_R = np.concatenate([det_p["right_ips"], det_d["right_ips"][cand]])
    all_fwhm = np.concatenate([det_p["fwhm_t"], det_d["fwhm_t"][cand]])
    is_prom = np.concatenate([np.ones(len(det_p["idx"]), bool),
                              np.zeros(len(cand), bool)])
    src = ([("p", j) for j in range(len(det_p["idx"]))] +
           [("d", cand[j]) for j in range(len(cand))])
    clusters = _cluster_peaks(all_idx, all_L, all_R, dt, cfg)

    pad = cfg.n_samples(cfg.fit_pad_min, dt, floor=1, odd=False)
    eff_noise = max(noise, cfg.noise_floor_frac * np.nanmax(y))
    accepted = []
    for cl in clusters:
        members = sorted(cl)
        defin = [m for m in members if is_prom[m]]
        tent = [m for m in members if not is_prom[m]]
        lo = int(max(0, np.floor(all_L[members].min()) - pad))
        hi = int(min(len(t) - 1, np.ceil(all_R[members].max()) + pad))
        xs, ys = t[lo:hi + 1], y[lo:hi + 1]
        # LOCAL resolution limit from this cluster's own peak widths
        loc_fwhm = np.median(all_fwhm[members]) if len(members) else 10 * dt
        min_sep_t = cfg.hybrid_min_sep_fwhm_frac * max(loc_fwhm, 2 * dt)

        def seed_of(m, _xlo=xs[0], _xhi=xs[-1]):
            kind, j = src[m]
            d = det_p if kind == "p" else det_d
            return _seed_params(t, d["idx"][j], d, j, dt, cfg,
                                xlo=_xlo, xhi=_xhi)

        chosen = list(defin)
        if not chosen and tent:                      # cluster of only shoulders
            tent_sorted = sorted(tent, key=lambda m: -y[all_idx[m]])
            chosen = [tent_sorted[0]]
            tent = tent_sorted[1:]
        tent = sorted(tent, key=lambda m: -y[all_idx[m]])[:cfg.hybrid_max_shoulders]
        sse_cur, _ = _fit_window_sse(xs, ys, [seed_of(m) for m in chosen], cfg) \
            if chosen else (float("inf"), None)
        N = len(xs)
        for m in tent:
            mt = t[all_idx[m]]
            if any(abs(mt - t[all_idx[c]]) < min_sep_t for c in chosen):
                continue
            trial = chosen + [m]
            sse_new, popt = _fit_window_sse(
                xs, ys, [seed_of(mm) for mm in trial], cfg)
            if popt is None or not np.isfinite(sse_new) or sse_new >= sse_cur:
                continue
            p_r, p_f = 4 * len(chosen), 4 * len(trial)
            if N - p_f <= 1:
                continue
            F = ((sse_cur - sse_new) / (p_f - p_r)) / (sse_new / (N - p_f))
            pval = f_dist.sf(F, p_f - p_r, N - p_f)
            rel = (sse_cur - sse_new) / max(sse_cur, 1e-12)
            A, mu, sig, tau = popt[-4:]
            comp_h = float(emg(xs, A, mu, sig, tau).max())
            if pval < cfg.hybrid_p_thresh and rel > cfg.hybrid_min_rel and \
               comp_h > cfg.hybrid_shoulder_snr * eff_noise:
                chosen = trial
                sse_cur = sse_new
        accepted.extend(chosen)

    final_idx, final = [], dict(t=[], height=[], fwhm_t=[],
                                left_ips=[], right_ips=[], prom=[])
    for m in sorted(accepted):
        kind, j = src[m]
        d = det_p if kind == "p" else det_d
        final_idx.append(int(d["idx"][j]))
        final["t"].append(float(d["t"][j]))
        final["height"].append(float(d["height"][j]))
        final["fwhm_t"].append(float(d["fwhm_t"][j]))
        final["left_ips"].append(float(d["left_ips"][j]))
        final["right_ips"].append(float(d["right_ips"][j]))
        final["prom"].append(float(d["prom"][j]) if "prom" in d
                             and j < len(d["prom"]) else np.nan)
    out = {k: np.array(v) for k, v in final.items()}
    out["idx"] = np.array(final_idx, dtype=int)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  8 · RESOLUTION / BASELINE-SEPARATION SCORECARD
# ══════════════════════════════════════════════════════════════════════════════

def _resolution_analysis(t, y, components, noise, cfg: PickerConfig):
    """Chromatographic resolution between adjacent peaks.

        Rs = (pos_b - pos_a) / (2 * (sigma_eff_a + sigma_eff_b))

    with the EMG effective width sigma_eff = sqrt(sigma^2 + tau^2).
    Rs >= cfg.rs_baseline is the textbook criterion for baseline separation.
    A model-free check is added: the valley between two apexes must fall close
    to baseline relative to the smaller peak.
    """
    n = len(components)
    if n == 0:
        return dict(pos=np.array([]), sigma_eff=np.array([]), Rs=np.array([]),
                    valley_ratio=np.array([]), pair_baseline=np.array([], bool),
                    separated=np.array([], bool), clusters=[],
                    n_separated=0, n_total=0)
    comp = sorted(components, key=lambda c: c["rt"])
    pos, sig_eff, height, apex_i = [], [], [], []
    for c in comp:
        tau = c["tau"] if np.isfinite(c.get("tau", np.nan)) else 0.0
        # position = Gaussian centre (monotonic with elution order); the tail is
        # folded into the effective width only, so adjacent gaps stay >= 0
        pos.append(c["rt"])
        sig_eff.append(np.sqrt(c["sigma"] ** 2 + tau ** 2))
        height.append(c["height"])
        apex_i.append(int(np.argmin(np.abs(t - c["rt"]))))
    pos = np.array(pos); sig_eff = np.array(sig_eff)
    height = np.array(height); apex_i = np.array(apex_i)

    Rs = np.full(n - 1, np.inf)
    valley_ratio = np.full(n - 1, 0.0)
    pair_baseline = np.zeros(n - 1, bool)
    for k in range(n - 1):
        denom = 2.0 * (sig_eff[k] + sig_eff[k + 1])
        Rs[k] = (pos[k + 1] - pos[k]) / denom if denom > 0 else np.inf
        a, b = apex_i[k], apex_i[k + 1]
        if b > a + 1:
            valley = float(np.min(y[a:b + 1]))
        else:
            valley = float(min(y[a], y[b]))
        hmin = max(min(height[k], height[k + 1]), 1e-9)
        valley_ratio[k] = max(valley, 0.0) / hmin
        pair_baseline[k] = (Rs[k] >= cfg.rs_baseline) or \
                           (valley_ratio[k] < cfg.valley_frac * 0.3)

    separated = np.ones(n, bool)
    for i in range(n):
        left = pair_baseline[i - 1] if i > 0 else True
        right = pair_baseline[i] if i < n - 1 else True
        separated[i] = left and right

    clusters = []
    start = 0
    for k in range(n - 1):
        if not pair_baseline[k]:
            continue
        if k >= start:
            size = k - start + 1
            if size >= 2:
                clusters.append(dict(i0=start, i1=k, size=size,
                                     t_start=float(pos[start]),
                                     t_end=float(pos[k]),
                                     span=float(pos[k] - pos[start]),
                                     min_Rs=float(np.min(Rs[start:k]))))
        start = k + 1
    if n - 1 >= start:
        size = (n - 1) - start + 1
        if size >= 2:
            clusters.append(dict(i0=start, i1=n - 1, size=size,
                                 t_start=float(pos[start]),
                                 t_end=float(pos[n - 1]),
                                 span=float(pos[n - 1] - pos[start]),
                                 min_Rs=float(np.min(Rs[start:n - 1]))))
    return dict(pos=pos, sigma_eff=sig_eff, Rs=Rs, valley_ratio=valley_ratio,
                pair_baseline=pair_baseline, separated=separated,
                clusters=clusters, n_separated=int(separated.sum()),
                n_total=n)


# ══════════════════════════════════════════════════════════════════════════════
#  9 · UCM ("hump") DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _split_region(t, y, a, b, medfs, cfg: PickerConfig):
    """Divide a hump [a,b] at internal valleys of the smoothed signal that dip
    more than cfg.hump_split_dip_frac below the smaller flanking sub-apex."""
    if b - a < cfg.hump_split_min_samples:
        return [(a, b)]
    seg = y[a:b + 1]
    order = max(int(medfs), 2)
    valleys = argrelmin(seg, order=order)[0]
    necks = []
    for m in valleys:
        left_apex = seg[:m].max() if m > 0 else seg[m]
        right_apex = seg[m:].max() if m < len(seg) - 1 else seg[m]
        flank = min(left_apex, right_apex)
        if flank > 0 and (flank - seg[m]) / flank >= cfg.hump_split_dip_frac:
            necks.append(a + m)
    bounds = [a] + necks + [b]
    subs = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if t[e] - t[s] >= cfg.hump_split_min_sub_span:
            subs.append((s, e))
    return subs or [(a, b)]


def _manual_hump_dict(t, y, t0, t1, cfg: PickerConfig):
    """Build a hump record for a user-specified [t0, t1] minute window, so
    every downstream consumer treats a manual hump exactly like an auto one."""
    sa = int(np.argmin(np.abs(t - min(t0, t1))))
    sb = int(np.argmin(np.abs(t - max(t0, t1))))
    sa, sb = min(sa, sb), max(sa, sb)
    if sb <= sa:
        sb = min(sa + 1, len(t) - 1)
    sub = y[sa:sb + 1]
    smax = float(np.max(sub)) if len(sub) else 0.0
    W = max(int(0.1 * len(sub)), cfg.min_window_samples)
    fl = float(np.median(minimum_filter1d(sub, size=W, mode="nearest"))) \
        if len(sub) else 0.0
    nz = max(float(np.median(sub)), 1e-9)
    prom = cfg.hump_wiggle_prom_frac * smax if smax > 0 else None
    return dict(
        t_start=float(t[sa]), t_end=float(t[sb]), span=float(t[sb] - t[sa]),
        n_wiggles=int(len(find_peaks(sub, prominence=prom)[0])),
        valley_floor_ratio=round(fl / nz, 3),
        hump_height=round(fl, 1),
        apex_t=float(t[sa + int(np.argmax(sub))]) if len(sub) else float(t[sa]),
        manual=True)


def _detect_unresolved_humps(t, y, noise, dt, cfg: PickerConfig, det_p=None,
                             windows_out: Optional[dict] = None):
    """Detect 'unresolved complex mixture' (UCM) regions - broad clusters where
    the signal never returns to baseline, so many low-prominence peaks ride on
    a raised floor.

    Detected from the signal's LOWER ENVELOPE (rolling minimum), not from the
    picked peaks - because peaks sitting on a hump have low prominence and are
    exactly what a prominence search under-counts.

    If cfg.hump_override is set, it REPLACES automatic detection entirely.

    RATE INVARIANCE.  Every window below is specified in MINUTES and converted
    through `cfg.n_samples`, exactly like the smoothing windows: a floor on the
    median peak width (`hump_min_fwhm_min`), a floor on the fallback width
    (`hump_fallback_floor_min`) and a floor on the lower-envelope window
    (`hump_envelope_min_min`).  A floor expressed in SAMPLES would be a long
    physical window on a slow export and a short one on a fast export, and the
    hump call would then depend on the export setting.

    The other half of rate invariance for this detector lives in
    `PickerConfig.effective_arpls_lam`: at fixed lam a decimated trace gets a
    much stiffer baseline, its corrected signal over a broad region sits
    higher, and the `valley_floor_ratio` gate below can flip on the export rate
    alone.  Both halves are needed.
    """
    if cfg.hump_override:
        return [_manual_hump_dict(t, y, a, b, cfg) for (a, b) in cfg.hump_override]

    snr = cfg.effective_hump_snr
    n = len(y)
    if n < 11:
        return []
    ymax = float(np.nanmax(y))
    if ymax <= 0:
        return []
    det = det_p if det_p is not None else _detect_peaks(t, y, noise, dt, cfg)
    if len(det["idx"]) >= 3:
        med_fwhm_min = max(float(np.median(det["fwhm_t"])), cfg.hump_min_fwhm_min)
    else:
        med_fwhm_min = max(float(cfg.hump_fallback_fwhm_min),
                           cfg.hump_fallback_floor_min)
    # sample counts derived from that ONE physical width
    med_fwhm_samp = cfg.n_samples(med_fwhm_min, dt, floor=1, odd=False)
    # lower envelope ~ floor under the wiggles (window ~ 2 peak widths)
    W = cfg.n_samples(max(cfg.hump_envelope_span_factor * med_fwhm_min,
                          cfg.hump_envelope_min_min), dt, floor=3, odd=True)
    if windows_out is not None:
        # Surfaced in PickerResult.windows so the report shows how many SAMPLES
        # the morphological envelope actually got.  Below ~9 samples the
        # erosion/dilation pair cannot be scaled continuously any more (a
        # +/-1 sample rounding is then a 20-30 % change in the net opening
        # reach), and the hump call becomes rate-sensitive on this trace.
        windows_out["hump_med_fwhm"] = med_fwhm_samp
        windows_out["hump_envelope"] = W
    lower = minimum_filter1d(y, size=W, mode="nearest")
    lower = maximum_filter1d(lower, size=med_fwhm_samp + 1, mode="nearest")
    # Hysteresis: a CORE level triggers a hump; a lower EXTEND level sets its
    # boundaries.
    hump_level = max(cfg.hump_core_noise_mult * noise, cfg.hump_level_frac * ymax)
    extend_level = max(cfg.hump_extend_noise_mult * noise,
                       cfg.hump_extend_frac * hump_level)
    core = lower > hump_level
    ext = lower > extend_level

    # --- seed regions that pass the hump gates -------------------------------
    seeds = []
    i = 0
    while i < n:
        if not ext[i]:
            i += 1
            continue
        j = i
        while j < n and ext[j]:
            j += 1
        a, b = i, j - 1
        if not np.any(core[a:b + 1]):
            i = j
            continue
        min_span = max(cfg.hump_min_span_factor * max(med_fwhm_min, 2 * dt),
                       cfg.hump_min_span_abs)
        if t[b] - t[a] >= min_span:
            seg = y[a:b + 1]
            real, _ = find_peaks(seg, prominence=max(
                snr * noise, cfg.hump_wiggle_prom_frac * float(np.max(seg))))
            floor_ratio = float(np.median(lower[a:b + 1])) / \
                max(float(np.median(seg)), 1e-9)
            if len(real) >= cfg.hump_min_real_peaks and \
               floor_ratio >= cfg.hump_min_floor_ratio:
                seeds.append((a, b))
        i = j + 1

    # --- optional grow: absorb adjacent LOW bumps across empty gaps ----------
    spans = [[t[a], t[b]] for (a, b) in seeds]
    merge_gap = cfg.hump_merge_gap
    if merge_gap and merge_gap > 0 and spans:
        bumps = []
        ii = 0
        while ii < n:
            if not ext[ii]:
                ii += 1
                continue
            jj = ii
            while jj < n and ext[jj]:
                jj += 1
            seg = y[ii:jj]
            if (t[jj - 1] - t[ii] >= 0.5 * max(med_fwhm_min, 2 * dt) and
                    float(np.max(seg)) < cfg.hump_grow_bump_hmax * ymax and
                    len(find_peaks(seg, prominence=max(
                        snr * noise,
                        cfg.hump_wiggle_prom_frac * float(np.max(seg))))[0]) >= 1):
                bumps.append([t[ii], t[jj - 1]])
            ii = jj

        def _gap_clean(p, q):
            m = (t > p) & (t < q)
            return (float(np.max(y[m])) if np.any(m) else 0.0) < \
                cfg.hump_grow_clean_frac * ymax

        changed = True
        while changed:
            changed = False
            for s in spans:
                for bp in bumps:
                    if bp[0] >= s[0] and bp[1] <= s[1]:
                        continue
                    if 0 <= bp[0] - s[1] < merge_gap and _gap_clean(s[1], bp[0]):
                        s[1] = max(s[1], bp[1]); changed = True
                    elif 0 <= s[0] - bp[1] < merge_gap and _gap_clean(bp[1], s[0]):
                        s[0] = min(s[0], bp[0]); changed = True
                    elif bp[0] < s[1] and bp[1] > s[0]:
                        s[0] = min(s[0], bp[0]); s[1] = max(s[1], bp[1])
                        changed = True
        spans.sort()
        merged = [spans[0]]
        for s in spans[1:]:
            if s[0] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], s[1])
            else:
                merged.append(s)
        spans = merged

    # --- build hump dicts (optionally split) --------------------------------
    def _hump_dict(sa, sb):
        sub = y[sa:sb + 1]
        fl = float(np.median(lower[sa:sb + 1]))
        return dict(
            t_start=float(t[sa]), t_end=float(t[sb]), span=float(t[sb] - t[sa]),
            n_wiggles=int(len(find_peaks(sub, prominence=max(
                snr * noise,
                cfg.hump_wiggle_prom_frac * float(np.max(sub))))[0])),
            valley_floor_ratio=round(fl / max(float(np.median(sub)), 1e-9), 3),
            hump_height=round(fl, 1),
            apex_t=float(t[sa + int(np.argmax(sub))]), manual=False)

    humps = []
    for (ts, te) in spans:
        sa = int(np.argmin(np.abs(t - ts)))
        sb = int(np.argmin(np.abs(t - te)))
        subranges = (_split_region(t, y, sa, sb, med_fwhm_samp, cfg)
                     if cfg.hump_split else [(sa, sb)])
        for (xa, xb) in subranges:
            humps.append(_hump_dict(xa, xb))
    return humps


# ══════════════════════════════════════════════════════════════════════════════
#  9b · TRACE HEALTH  (drift diagnostics - NEVER an input to compute_crf)
# ══════════════════════════════════════════════════════════════════════════════

#: Every key of the trace-health block, in report order.  Consumers (the BO
#: drift monitor in 03b) iterate this rather than hardcoding names.
TRACE_HEALTH_KEYS = (
    "trace_dt_min", "trace_pts_per_min", "trace_n_points", "trace_tmax_min",
    "trace_baseline_offset", "trace_signal_span", "trace_max_signal",
    "trace_total_area", "trace_noise_sd", "trace_first_peak_rt",
    "trace_last_peak_rt", "trace_mean_peak_width_min",
    "trace_median_peak_width_min", "trace_mean_tailing",
    "trace_file_mtime_iso",
)

#: The subset that is numeric (the rest is provenance metadata).
TRACE_HEALTH_NUMERIC_KEYS = tuple(k for k in TRACE_HEALTH_KEYS
                                  if k != "trace_file_mtime_iso")


def usp_tailing(A, mu, sigma, tau):
    """USP tailing factor T = (a + b) / (2a) at 5 % of peak height.

    Computed from the FITTED EMG rather than from the raw trace, so it stays
    defined where peaks overlap.  T = 1 is a symmetric peak; T > 1 tails.
    Returns 1.0 for the unfitted fallback components, which carry tau = nan and
    therefore no shape information.
    """
    if not np.isfinite(tau) or tau <= 0 or not np.isfinite(sigma):
        return 1.0
    sig, ta = abs(sigma), abs(tau)
    x = np.linspace(mu - (8 * sig + 12 * ta), mu + (8 * sig + 12 * ta), 2001)
    y = emg(x, A, mu, sig, ta)
    if y.max() <= 0:
        return 1.0
    ap = int(np.argmax(y))
    lvl = 0.05 * y[ap]
    l = x[:ap + 1][y[:ap + 1] >= lvl]
    r = x[ap:][y[ap:] >= lvl]
    if l.size == 0 or r.size == 0:
        return 1.0
    a = x[ap] - l[0]
    b = r[-1] - x[ap]
    return float((a + b) / (2 * a)) if a > 1e-12 else 1.0


def _emg_fwhm_min(c):
    """Width of one component in MINUTES.

    For a FITTED EMG the Gaussian and exponential contributions add in
    quadrature, so the effective standard deviation is sqrt(sigma^2 + tau^2)
    and the width quoted is 2*sqrt(2 ln 2) = 2.355 times it - the same
    sigma_eff the resolution scorecard uses, so the two cannot disagree.  For
    an unfitted fallback component tau is nan and sigma was itself derived
    from the measured FWHM, so this returns that FWHM back unchanged.
    """
    sig = float(c.get("sigma", np.nan))
    tau = float(c.get("tau", np.nan))
    if not np.isfinite(sig):
        return float("nan")
    if not np.isfinite(tau):
        tau = 0.0
    return 2.355 * float(np.sqrt(sig ** 2 + tau ** 2))


def _file_mtime_iso(path: Optional[str]) -> str:
    """The trace file's modification time as an ISO-8601 string.

    THIS IS THE FILE'S mtime, NOT THE INSTRUMENT'S ACQUISITION TIME.  It is a
    timestamp PROXY, offered because instrument exports routinely carry no run
    timestamp at all.  It is wrong whenever the files were copied, exported in
    a batch, moved between
    filesystems, restored from a backup or synced by a cloud client - all of
    which set the mtime to the copy time.  Use it to CHECK a recorded run
    order, never to establish one.  Empty string when `analyse()` was given
    arrays rather than a path, or when the file has since gone.
    """
    if not path:
        return ""
    try:
        return _datetime.datetime.fromtimestamp(
            os.path.getmtime(path)).isoformat(timespec="seconds")
    except OSError:
        return ""


def _trace_health(path, t, y_raw, sm, noise, dt, pts_per_min,
                  det_h, components) -> Dict[str, Any]:
    """Trace-level health / drift descriptors for ONE analysed trace.

    DIAGNOSTICS ONLY.  Nothing here reaches `compute_crf`: `CRF_COLUMNS` is
    still exactly ["n_clean_peaks", "hump_time_fraction"], and
    `measurements_dict()` - the handoff that actually feeds the objective -
    never looks at this block.  The point of these numbers is to answer "did
    the INSTRUMENT change between runs", which is a question about the trace,
    not about the separation.

    Definitions (frozen - the drift monitor codes against them):
      trace_dt_min                median sample spacing, minutes
      trace_pts_per_min           1 / trace_dt_min
      trace_n_points              samples in the file
      trace_tmax_min              last retention time
      trace_baseline_offset       min of the RAW signal, before any baseline
                                  correction - the detector's zero offset
      trace_signal_span           raw max - raw min
      trace_max_signal            max of the smoothed, baseline-corrected trace
      trace_total_area            trapezoidal integral of that same trace
      trace_noise_sd              the estimator's sigma (cfg.noise_override
                                  when one was supplied)
      trace_first_peak_rt         first / last picked peak (ALL categories, so
      trace_last_peak_rt          this differs from the clean-only
                                  first_peak_rt / last_peak_rt in the feature
                                  row); nan when nothing was picked
      trace_mean_peak_width_min   mean / median component width, from the EMG
      trace_median_peak_width_min fit where one succeeded, else the measured
                                  FWHM (see `_emg_fwhm_min`)
      trace_mean_tailing          mean USP tailing factor over the components
      trace_file_mtime_iso        see `_file_mtime_iso` - a PROXY, and it lies
                                  if the files were copied
    """
    tp = np.asarray(det_h.get("t", np.array([])), dtype=float)
    widths = np.array([_emg_fwhm_min(c) for c in components], dtype=float) \
        if components else np.array([])
    widths = widths[np.isfinite(widths)]
    tails = np.array([usp_tailing(c.get("area", np.nan), c["rt"],
                                  c.get("sigma", np.nan), c.get("tau", np.nan))
                      for c in components], dtype=float) if components \
        else np.array([])
    tails = tails[np.isfinite(tails)]
    if len(widths) == 0 and len(tp):
        # deconvolution off / no components: fall back to the measured FWHM
        widths = np.asarray(det_h.get("fwhm_t", np.array([])), dtype=float)
        widths = widths[np.isfinite(widths)]
    return {
        "trace_dt_min": float(dt),
        "trace_pts_per_min": float(pts_per_min),
        "trace_n_points": int(len(t)),
        "trace_tmax_min": float(t[-1]) if len(t) else float("nan"),
        "trace_baseline_offset": float(np.nanmin(y_raw)) if len(y_raw)
        else float("nan"),
        "trace_signal_span": float(np.nanmax(y_raw) - np.nanmin(y_raw))
        if len(y_raw) else float("nan"),
        "trace_max_signal": float(np.nanmax(sm)) if len(sm) else float("nan"),
        "trace_total_area": float(np.trapezoid(sm, t)) if len(t) > 1 else 0.0,
        "trace_noise_sd": float(noise),
        "trace_first_peak_rt": float(np.min(tp)) if len(tp) else float("nan"),
        "trace_last_peak_rt": float(np.max(tp)) if len(tp) else float("nan"),
        "trace_mean_peak_width_min": float(np.mean(widths)) if len(widths)
        else float("nan"),
        "trace_median_peak_width_min": float(np.median(widths)) if len(widths)
        else float("nan"),
        "trace_mean_tailing": float(np.mean(tails)) if len(tails)
        else float("nan"),
        "trace_file_mtime_iso": _file_mtime_iso(path),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  10 · THE RESULT OBJECT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PickerResult:
    """Everything one `analyse()` pass produced.  Formatters read this and only
    this; nothing downstream may recompute."""
    # provenance
    path: Optional[str]
    name: str
    cfg: PickerConfig
    # trace & stages
    t: np.ndarray
    y_raw: np.ndarray
    baseline: np.ndarray
    corrected: np.ndarray
    smoothed: np.ndarray
    noise: float
    dt: float
    pts_per_min: float
    # window conversion actually used (minutes -> samples), for the report
    windows: Dict[str, int]
    # detections
    detection_prominence: Dict[str, np.ndarray]
    detection_hybrid: Dict[str, np.ndarray]
    # categorisation (aligned to detection_hybrid)
    categories: np.ndarray
    n_clean: int
    n_shoulder: int
    n_hump: int
    # humps
    humps: List[dict]
    # EMG
    components: List[dict]
    model: np.ndarray
    # resolution
    resolution: dict          # over the EMG components
    resolution_clean: dict    # over clean prominence peaks (feeds the features)
    clean_components: List[dict]
    # method-level features
    features: Dict[str, Any]
    # trace-level health / drift diagnostics (see TRACE_HEALTH_KEYS).  These
    # are ALSO merged into `features`, so extract_features() surfaces them;
    # they are deliberately absent from CRF_COLUMNS and MEASUREMENT_KEYS.
    trace_health: Dict[str, Any]

    # convenience passthroughs
    @property
    def detection(self):
        return self.detection_hybrid

    def __getitem__(self, k):
        """Dict-style access: r['t'], r['smoothed'], r['detection']."""
        if k == "detection":
            return self.detection_hybrid
        return getattr(self, k)


# ══════════════════════════════════════════════════════════════════════════════
#  11 · THE ONE PIPELINE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def _as_trace(path_or_arrays):
    """Accept a path, a (t, y) pair, or an (N,2) array.  Returns (path, name,
    t, y)."""
    if isinstance(path_or_arrays, PickerResult):
        return (path_or_arrays.path, path_or_arrays.name,
                path_or_arrays.t, path_or_arrays.y_raw)
    if isinstance(path_or_arrays, (str, bytes, os.PathLike)):
        p = os.fspath(path_or_arrays)
        t, y = load_chromatogram(p)
        return p, os.path.basename(p), t, y
    if isinstance(path_or_arrays, (tuple, list)) and len(path_or_arrays) == 2:
        t = np.asarray(path_or_arrays[0], dtype=float)
        y = np.asarray(path_or_arrays[1], dtype=float)
        return None, "<arrays>", t, y
    arr = np.asarray(path_or_arrays, dtype=float)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return None, "<arrays>", arr[:, 0], arr[:, 1]
    raise TypeError("analyse() expects a path, (t, y), or an (N,2) array")


def analyse(path_or_arrays, cfg: PickerConfig = DEFAULT_CONFIG,
            name: Optional[str] = None) -> PickerResult:
    """Run the ENTIRE pipeline exactly once and return everything.

        load -> arPLS baseline -> clip negatives -> Savitzky-Golay smooth ->
        prominence detection -> fit-gated second-derivative shoulders ->
        UCM hump detection -> categorisation -> EMG deconvolution ->
        resolution scorecard -> method-level features

    Every threshold comes from `cfg`.  There are no globals and no defaults
    buried in call sites: this is the only place the stages are wired together.
    """
    path, auto_name, t, y_raw = _as_trace(path_or_arrays)
    if name is not None:
        auto_name = name

    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    pts_per_min = 1.0 / dt if dt > 0 else float("nan")

    # --- stages 1-3: baseline, noise, clip, smooth ---------------------------
    corr, bkg = _correct_baseline(t, y_raw, cfg, dt=dt)
    noise = (estimate_noise(corr, dt, cfg) if cfg.noise_override is None
             else float(cfg.noise_override))   # from the PRE-clip residual
    if cfg.clip_negative:
        corr = np.clip(corr, 0.0, None)
    sm = _smooth(corr, dt, cfg)
    if cfg.clip_negative:
        # Savitzky-Golay can ring (undershoot) around sharp tall peaks
        sm = np.clip(sm, 0.0, None)

    windows = dict(
        sg_window=cfg.n_samples(cfg.sg_window_min, dt, poly=cfg.sg_poly,
                                n_total=len(sm)),
        min_width=cfg.width_samples(dt),
        d2_window=cfg.n_samples(cfg.d2_window_min, dt, poly=cfg.d2_poly,
                                n_total=len(sm)),
        merge_min_sep=cfg.n_samples(cfg.merge_min_sep_min, dt, floor=1, odd=False),
        cluster_gap=cfg.n_samples(cfg.cluster_gap_min, dt, floor=0, odd=False),
        fit_pad=cfg.n_samples(cfg.fit_pad_min, dt, floor=1, odd=False),
    )

    # --- stage 4/5: detection (computed ONCE, shared by everything) ----------
    det_p = _detect_peaks(t, sm, noise, dt, cfg)
    if cfg.detection == "prominence":
        det_d, det_h = _empty_detection(), det_p
    elif cfg.detection == "second_derivative":
        det_d = _detect_peaks_2d(t, sm, noise, dt, cfg)
        det_h = det_d
    elif cfg.detection == "hybrid":
        det_d = _detect_peaks_2d(t, sm, noise, dt, cfg)
        det_h = _merge_detections(t, det_p, det_d, dt, cfg)
    elif cfg.detection == "hybrid_fit":
        det_d = _detect_peaks_2d(t, sm, noise, dt, cfg)
        det_h = _detect_hybrid_fitgated(t, sm, noise, dt, cfg,
                                        det_p=det_p, det_d=det_d)
    else:
        raise ValueError("unknown cfg.detection %r" % (cfg.detection,))

    # --- stage 8: humps (reuse the SAME prominence detection) ---------------
    humps = _detect_unresolved_humps(t, sm, noise, dt, cfg, det_p=det_p,
                                     windows_out=windows)

    # --- stage 10: categorisation -------------------------------------------
    P = set(int(i) for i in det_p["idx"].tolist())

    def in_hump(tp):
        return any(h["t_start"] <= tp <= h["t_end"] for h in humps)

    cats = []
    for k, ix in enumerate(det_h["idx"]):
        tp = det_h["t"][k]
        cats.append("hump" if in_hump(tp)
                    else ("clean" if int(ix) in P else "shoulder"))
    cats = np.array(cats)
    n_clean = int(np.sum(cats == "clean"))
    n_sh = int(np.sum(cats == "shoulder"))
    n_hp = int(np.sum(cats == "hump"))

    # --- stage 6: EMG deconvolution on the hybrid detection ------------------
    comps: List[dict] = []
    model = np.zeros_like(sm)
    if cfg.deconvolve and len(det_h["idx"]):
        comps, model_raw = _deconvolve_emg(t, sm, det_h, dt, cfg)
        model = _reconstruct_model(t, comps) if comps else model_raw
    comps = sorted(comps, key=lambda c: c["rt"])

    # --- stage 7: resolution scorecards --------------------------------------
    res_emg = _resolution_analysis(t, sm, comps, noise, cfg)

    # feature-side resolution: over the CLEAN (off-hump) PROMINENCE peaks,
    # as pseudo-Gaussian components (tau = nan).  This is what the master sheet
    # records; it is deliberately independent of whether deconvolution ran.
    clean_comp = []
    for k in range(len(det_p["idx"])):
        if not in_hump(det_p["t"][k]):
            clean_comp.append(dict(rt=float(det_p["t"][k]),
                                   sigma=float(det_p["fwhm_t"][k] / 2.355),
                                   tau=float("nan"),
                                   height=float(det_p["height"][k])))
    res_clean = _resolution_analysis(t, sm, clean_comp, noise, cfg)

    # --- method-level features (computed HERE, formatted elsewhere) ----------
    total_time = float(t[-1] - t[0])
    hump_spans = [h["t_end"] - h["t_start"] for h in humps]
    total_hump = float(sum(hump_spans))
    longest_hump = float(max(hump_spans)) if hump_spans else 0.0

    Rs = res_clean["Rs"][np.isfinite(res_clean["Rs"])] \
        if len(res_clean["Rs"]) else np.array([])
    Rs_c = np.clip(Rs, 0, cfg.rs_clip)
    sum_rs = float(np.sum(Rs_c)) if len(Rs_c) else 0.0
    mean_rs = float(np.mean(Rs_c)) if len(Rs_c) else 0.0
    clean_rts = [c["rt"] for c in clean_comp]

    features = dict(
        method=auto_name.replace(".TXT", "") if auto_name else "<arrays>",
        total_time_min=round(total_time, 3),
        n_clean_peaks=n_clean,
        n_shoulder_fronting=n_sh,
        n_on_hump_peaks=n_hp,
        n_total_picked=int(len(cats)),
        n_ucm_humps=len(humps),
        total_hump_time_min=round(total_hump, 3),
        longest_hump_min=round(longest_hump, 3),
        hump_time_fraction=round(total_hump / total_time, 4) if total_time else 0.0,
        n_baseline_separated=int(res_clean["n_separated"]),
        frac_baseline_sep=round(res_clean["n_separated"] / n_clean, 3)
        if n_clean else 0.0,
        sum_resolution=round(sum_rs, 3),
        mean_resolution=round(mean_rs, 3),
        first_peak_rt=round(min(clean_rts), 3) if clean_rts else np.nan,
        last_peak_rt=round(max(clean_rts), 3) if clean_rts else np.nan,
        noise=round(float(noise), 4),
        max_signal=round(float(np.nanmax(sm)), 1),
    )

    # --- stage 11: trace health (DIAGNOSTIC ONLY - see _trace_health) --------
    health = _trace_health(path, t, y_raw, sm, noise, dt, pts_per_min,
                           det_h, comps)
    assert not (set(health) & set(CRF_COLUMNS)), \
        "a trace-health key collided with a CRF input"
    features.update(health)

    return PickerResult(
        path=path, name=auto_name, cfg=cfg,
        t=t, y_raw=y_raw, baseline=bkg, corrected=corr, smoothed=sm,
        noise=float(noise), dt=dt, pts_per_min=pts_per_min, windows=windows,
        detection_prominence=det_p, detection_hybrid=det_h,
        categories=cats, n_clean=n_clean, n_shoulder=n_sh, n_hump=n_hp,
        humps=humps, components=comps, model=model,
        resolution=res_emg, resolution_clean=res_clean,
        clean_components=clean_comp, features=features,
        trace_health=health)


# ══════════════════════════════════════════════════════════════════════════════
#  12 · PURE FORMATTERS  (no recomputation - by construction they agree)
# ══════════════════════════════════════════════════════════════════════════════

CHROMATOGRAPHIC_FEATURE_KEYS = [
    "method", "total_time_min", "n_clean_peaks", "n_shoulder_fronting",
    "n_on_hump_peaks", "n_total_picked", "n_ucm_humps", "total_hump_time_min",
    "longest_hump_min", "hump_time_fraction", "n_baseline_separated",
    "frac_baseline_sep", "sum_resolution", "mean_resolution",
    "first_peak_rt", "last_peak_rt", "noise", "max_signal",
]

#: The chromatographic feature keys, with the trace-health block appended.
#: Notebooks 03b / 04 / 05 and the master sheet index rows by these names, so
#: the names are frozen.
FEATURE_KEYS = CHROMATOGRAPHIC_FEATURE_KEYS + list(TRACE_HEALTH_KEYS)


def extract_features(result_or_path, cfg: PickerConfig = DEFAULT_CONFIG) -> dict:
    """Method-level feature row for the master sheet.

    Key names are FROZEN - notebooks 03b / 04 / 05 and the master sheet depend
    on them.  This is a pure formatter: if given a `PickerResult` it does no
    computation whatsoever; if given a path it calls `analyse()` once.

    The row ends with the `trace_*` health block (`TRACE_HEALTH_KEYS`).  Those
    are diagnostics for the drift monitor; `compute_crf` does not read them and
    `CRF_COLUMNS` does not contain them.
    """
    res = result_or_path if isinstance(result_or_path, PickerResult) \
        else analyse(result_or_path, cfg)
    return {k: res.features[k] for k in FEATURE_KEYS}


def _feature_rows(ft: dict):
    """Ordered (label, value) feature pairs for the report panel (no CRF)."""
    return [
        ("Total run time (min)", f"{ft['total_time_min']:.2f}"),
        ("Clean peaks", ft["n_clean_peaks"]),
        ("Shoulder / fronting", ft["n_shoulder_fronting"]),
        ("On-hump peaks", ft["n_on_hump_peaks"]),
        ("UCM humps", ft["n_ucm_humps"]),
        ("Total hump time (min)", f"{ft['total_hump_time_min']:.2f}"),
        ("Longest hump (min)", f"{ft['longest_hump_min']:.2f}"),
        ("Hump time fraction", f"{ft['hump_time_fraction']:.1%}"),
        ("Baseline-separated", f"{ft['n_baseline_separated']}/{ft['n_clean_peaks']}"),
        ("Sum resolution (clean)", f"{ft['sum_resolution']:.2f}"),
        ("Avg resolution (clean)", f"{ft['mean_resolution']:.2f}"),
        ("Last eluting (min)", f"{ft['last_peak_rt']}"),
    ]


def feature_report(result: PickerResult,
                   cfg: Optional[PickerConfig] = None) -> str:
    """Plain-ASCII report for ONE analysed trace.

    Reads `result.features` verbatim, so the report and the numbers recorded to
    the master sheet are the SAME numbers - they cannot disagree.
    """
    if cfg is None:
        cfg = result.cfg
    ft = result.features
    W = 78
    L = []
    L.append("=" * W)
    L.append("PEAK PICKING REPORT   %s" % ft["method"])
    L.append("=" * W)

    L.append("")
    L.append("CONFIG (single source of truth)")
    L.extend(cfg.summary_lines())

    L.append("")
    L.append("TRACE")
    L.append("  %-26s %s" % ("samples", len(result.t)))
    L.append("  %-26s %.4f - %.4f min" % ("time range", result.t[0], result.t[-1]))
    L.append("  %-26s %.6f min  (%.2f pts/min)"
             % ("median dt", result.dt, result.pts_per_min))
    L.append("  %-26s %.4f" % ("noise sigma (time-lag)", result.noise))
    L.append("  %-26s %.1f" % ("max signal (processed)", ft["max_signal"]))
    L.append("  %-26s %g" % ("arPLS lam applied",
                             cfg.effective_arpls_lam(result.dt)))

    L.append("")
    L.append("WINDOWS  (minutes -> samples for THIS trace's dt)")
    L.append("  %-22s %10s %10s %10s" % ("window", "minutes", "samples", "@60pts"))
    for label, key, minutes, at60 in [
            ("savitzky-golay", "sg_window", cfg.sg_window_min, 11),
            ("min peak width", "min_width", cfg.min_width_min, 3),
            ("2nd-deriv window", "d2_window", cfg.d2_window_min, 21),
            ("merge separation", "merge_min_sep", cfg.merge_min_sep_min, 4),
            ("cluster gap", "cluster_gap", cfg.cluster_gap_min, 2),
            ("fit padding", "fit_pad", cfg.fit_pad_min, 5)]:
        L.append("  %-22s %10.4f %10.2f %10d"
                 % (label, minutes, result.windows[key], at60))
    if "hump_envelope" in result.windows:
        L.append("  %-22s %10s %10.2f %10s"
                 % ("hump envelope", "auto", result.windows["hump_envelope"],
                    "-"))
        if result.windows["hump_envelope"] < 9:
            L.append("      ^ fewer than 9 samples: the hump call is "
                     "rate-sensitive on this trace")

    L.append("")
    L.append("DETECTION")
    L.append("  %-26s %d" % ("prominence picks",
                             len(result.detection_prominence["idx"])))
    L.append("  %-26s %d" % ("hybrid (fit-gated) picks",
                             len(result.detection_hybrid["idx"])))
    L.append("  %-26s clean=%d  shoulder/fronting=%d  on-hump=%d"
             % ("categories", ft["n_clean_peaks"], ft["n_shoulder_fronting"],
                ft["n_on_hump_peaks"]))

    L.append("")
    L.append("UCM HUMPS  (%d)" % ft["n_ucm_humps"])
    if result.humps:
        L.append("  %3s %9s %9s %8s %8s %10s %8s"
                 % ("#", "start", "end", "span", "wiggles", "floor_rat", "mode"))
        for i, h in enumerate(result.humps, 1):
            L.append("  %3d %9.3f %9.3f %8.3f %8d %10.3f %8s"
                     % (i, h["t_start"], h["t_end"], h["span"], h["n_wiggles"],
                        h["valley_floor_ratio"],
                        "manual" if h.get("manual") else "auto"))
    else:
        L.append("  (none)")

    L.append("")
    L.append("RESOLUTION SCORECARD")
    ra = result.resolution
    L.append("  %-26s %d / %d" % ("EMG comps baseline-sep",
                                  ra["n_separated"], ra["n_total"]))
    if ra["clusters"]:
        for c in ra["clusters"]:
            L.append("    unresolved cluster  %6.2f - %6.2f min   %d peaks  "
                     "min Rs=%.2f"
                     % (c["t_start"], c["t_end"], c["size"], c["min_Rs"]))
    else:
        L.append("    (no unresolved EMG clusters)")
    rc = result.resolution_clean
    L.append("  %-26s %d / %d" % ("clean peaks baseline-sep",
                                  rc["n_separated"], rc["n_total"]))

    L.append("")
    L.append("METHOD FEATURES  (exactly what extract_features() returns)")
    for k, v in _feature_rows(ft):
        L.append("  %-26s %12s" % (k, v))

    L.append("")
    L.append("TRACE HEALTH  (drift diagnostics - NOT inputs to the CRF)")
    for k in TRACE_HEALTH_KEYS:
        v = result.trace_health[k]
        if isinstance(v, str):
            L.append("  %-28s %s" % (k, v if v else "(no file)"))
        elif isinstance(v, (int, np.integer)):
            L.append("  %-28s %12d" % (k, v))
        else:
            L.append("  %-28s %12.4f" % (k, v))

    L.append("")
    L.append("MEASUREMENTS HANDOFF  (for notebook 03 / 05)")
    meas = measurements_dict(result)
    for k in MEASUREMENT_KEYS:
        if k in meas:
            L.append("  %-26s %s" % (k, meas[k]))
    L.append("  %-26s %.3f" % ("CRF", compute_crf(meas)))
    L.append("=" * W)
    return "\n".join(L)


# ══════════════════════════════════════════════════════════════════════════════
#  13 · SHARED BLOCK · CRF DEFINITION
# ══════════════════════════════════════════════════════════════════════════════
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SHARED BLOCK · CRF DEFINITION — keep identical in notebooks 01/03/04/05   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# The Chromatographic Response Function is the single score the whole workflow
# optimises. It is PLUGGABLE: edit compute_crf() here and paste the same block
# into the other notebooks — recording, the forward model and the explorer all
# pick it up automatically.
#
#   CRF_COLUMNS : the raw measurements compute_crf() consumes. Notebook 01
#                 extracts them from the chromatogram.
#
#       CRF = n_clean_peaks * (1 - hump_time_fraction) ** 2
#
# A high count of well-separated peaks is good; time spent under an
# unresolved-complex-mixture hump is bad, and the square makes the penalty bite
# only once the hump occupies a real fraction of the run.
#
# WHY THIS OBJECTIVE. 56 pre-declared candidates (34 hand-written + 22 from the
# literature) were scored on 20 expert-ranked chromatograms against BOTH of two
# reviewers' orderings and one reviewer's 0-10 ratings. This one finished 1st of
# 56 on the composite and 1st of 56 on tau vs the reviewers, Spearman vs the
# ratings, tie-aware tau vs the ratings, out-of-sample isotonic calibration and
# NDCG@20.
#
#     Kendall tau vs the two reviewers   0.857   (inter-rater ceiling 0.937 = 91.5 %)
#     Spearman vs the 0-10 ratings       0.970
#     Pearson  vs the 0-10 ratings       0.948
#     isotonic LOO calibration RMSE      0.734 rating points   (best of 56)
#     NDCG@20 / NDCG@5                   0.997 / 0.994
#     picker-config sensitivity          0.200 sd of the set   (rank 17/55, ok)
#
#   0 of 56 candidates beat it on tau, on Spearman-vs-rating or on calibration.
#   0 of 75 candidate x metric bootstrap comparisons put a challenger ahead with
#   a 95 % interval excluding zero; 0 survive Bonferroni (x280). At n = 20 the
#   leading group is not resolvable, and this objective is at the top of it.
#
# THE OBJECTIVE IS FIXED, on purpose. A moving objective makes every campaign
# incomparable with every other, so it is settled here and not re-derived
# downstream.
#
# The trace-health block (TRACE_HEALTH_KEYS) is NOT in CRF_COLUMNS and never
# will be. It describes the instrument, not the separation; feeding it to the
# objective would let a drifting detector move the score the optimiser is
# climbing.

RS_BASELINE = 1.5          # textbook baseline-resolution target (descriptive only)

CRF_COLUMNS = ["n_clean_peaks", "hump_time_fraction"]

# recorded by the picker but NOT part of the objective — kept for diagnostics
CRF_DIAGNOSTIC_COLUMNS = ["sum_resolution", "mean_resolution", "frac_baseline_sep",
                          "n_shoulder_fronting", "n_ucm_humps", "sum_soft_resolution"]


def compute_crf(row):
    """Score one method (or a whole table) from its raw measurements.

    CRF = n_clean_peaks * (1 - hump_time_fraction) ** 2

    `row` may be a dict, a pandas Series or a pandas DataFrame containing
    CRF_COLUMNS; the return type follows the input.

    A high count of well-separated peaks is good; time spent under an
    unresolved-complex-mixture hump is bad, and the square makes the penalty
    bite only once the hump occupies a real fraction of the run.
    """
    return row["n_clean_peaks"] * (1.0 - row["hump_time_fraction"]) ** 2


MEASUREMENT_KEYS = ["n_clean_peaks", "hump_time_fraction", "n_ucm_humps",
                    "total_hump_time_min", "longest_hump_min",
                    "total_time_min"]

# The trace-health block is deliberately NOT here.  It is diagnostic: the drift
# monitor reads it from `extract_features()` / `PickerResult.trace_health` and
# charts it, but it must never enter the objective, or a drifting instrument
# would start moving the score the optimiser is climbing.
assert not (set(MEASUREMENT_KEYS) & set(TRACE_HEALTH_KEYS))
assert not (set(CRF_COLUMNS) & set(TRACE_HEALTH_KEYS))


def measurements_dict(result_or_features) -> dict:
    """The exact dict notebooks 03 §8 / 05 §9 record into the master sheet."""
    ft = (result_or_features.features
          if isinstance(result_or_features, PickerResult)
          else result_or_features)
    meas = {k: ft[k] for k in MEASUREMENT_KEYS if k in ft}
    missing = [k for k in CRF_COLUMNS if k not in meas]
    assert not missing, f"feature extraction did not produce {missing}"
    return meas


#: EVERY chromatographic feature a recorded RUN should carry, plus the whole
#: trace-health block.  A strict SUPERSET of MEASUREMENT_KEYS.
#:
#: MEASUREMENT_KEYS is the minimal handoff: the two columns the objective reads
#: plus the four hump descriptors the sheet has always carried.  It is not the
#: right set to RECORD.  `analyse()` produces seventeen chromatographic features
#: and fifteen trace-health descriptors for the price of one pipeline pass, and
#: a run whose trace is later moved, overwritten or re-exported cannot be
#: re-measured.  Recording the wide row makes the campaign re-analysable -
#: correlating shoulder counts against temperature, re-fitting a different
#: objective on the same runs, or explaining a drift verdict - without touching
#: a single .TXT again.
#:
#: This changes NOTHING about the objective.  `compute_crf` still reads exactly
#: CRF_COLUMNS, and the asserts below are the guard that keeps it that way.
RECORD_FEATURE_KEYS = [k for k in CHROMATOGRAPHIC_FEATURE_KEYS if k != "method"]
RECORD_KEYS = RECORD_FEATURE_KEYS + list(TRACE_HEALTH_KEYS)

assert not (set(CRF_COLUMNS) - set(RECORD_KEYS)), \
    "the objective's own inputs must be recorded"
assert not (set(MEASUREMENT_KEYS) - set(RECORD_KEYS)), \
    "RECORD_KEYS must be a superset of the minimal handoff"
assert not (set(CRF_COLUMNS) & set(TRACE_HEALTH_KEYS)), \
    "trace health must stay out of the objective"


def record_dict(result_or_features) -> dict:
    """The full row a recorded RUN should carry: every chromatographic feature
    plus the trace-health block.

    Use this - not `measurements_dict()` - anywhere a run is written to the
    master sheet.  `measurements_dict()` is kept unchanged for callers that
    only want the objective's own inputs.

    Keys absent from the source dict are simply omitted, so a features dict
    produced by an older picker still round-trips.
    """
    ft = (result_or_features.features
          if isinstance(result_or_features, PickerResult)
          else result_or_features)
    out = {k: ft[k] for k in RECORD_KEYS if k in ft}
    missing = [k for k in CRF_COLUMNS if k not in out]
    assert not missing, f"feature extraction did not produce {missing}"
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  14 · CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(0)
    for _p in sys.argv[1:]:
        _res = analyse(_p, DEFAULT_CONFIG)
        print(feature_report(_res, DEFAULT_CONFIG))
        print()
