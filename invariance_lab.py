"""
invariance_lab.py — Pocket CERN Collider · Lead-Scientist analysis engine
=========================================================================

* **Reconstruction** — builds signal candidates (ℓ⁺ℓ⁻, γγ, 4ℓ, eμ) from the *smeared*
  detector objects of ``detector_sim`` and computes the Lorentz-invariant mass

      M_inv = √((Σ E)² − |Σ p⃗|²)

  comparing it with the Monte-Carlo truth mass.
* **Luminosity runs** — mixes signal and continuum-background events, histograms
  M_inv and fits *peak ⊗ resolution + exponential background* with
  ``scipy.optimize.curve_fit``, extracting M_fit ± σ_M, Γ_fit, the resolution and the
  local significance S/√B.
* **Mystery Event Challenge** — serves blind detector events from a pool of iconic
  signatures, publishes only detector-level clues, and grades the player's guess of the
  final-state particles and of the parent channel.
"""

from __future__ import annotations

import itertools
import math
import sys
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import voigt_profile

import detector_sim as ds
import physics_engine as pe
from detector_sim import DetectedObject, DetectorConfig, DetectorEvent, Layer
from physics_engine import FourVector

# --------------------------------------------------------------------------------------
# Invariant mass & candidates
# --------------------------------------------------------------------------------------


def invariant_mass(objects: Iterable[DetectedObject | FourVector]) -> float:
    """M_inv = √((ΣE)² − |Σp⃗|²) (compensated sums; 0 for numerically space-like sums)."""
    vs = [o.p4 if isinstance(o, DetectedObject) else o for o in objects]
    tot = FourVector.sum(vs)
    return math.sqrt(max(tot.mass2, 0.0))


@dataclass(frozen=True)
class Selection:
    lepton_min_pt: float = 0.5
    photon_min_pt: float = 2.0
    diphoton_lead_pt: float = 10.0  # for automatic final-state choice
    z_mass: float = pe.pdg(pe.Z0).mass


@dataclass
class Candidate:
    final_state: str  # mumu | ee | gammagamma | 4l | emu
    objects: list[DetectedObject]
    mass: float
    truth_mass: float | None = None
    sub_masses: tuple[float, ...] = ()  # (m_Z1, m_Z2) for 4ℓ

    @property
    def p4(self) -> FourVector:
        return FourVector.sum(o.p4 for o in self.objects)

    @property
    def residual(self) -> float | None:
        return None if self.truth_mass is None else self.mass - self.truth_mass

    def label(self) -> str:
        return {"mumu": "μ⁺μ⁻", "ee": "e⁺e⁻", "gammagamma": "γγ", "4l": "4ℓ",
                "emu": "e±μ∓"}.get(self.final_state, self.final_state)


def _os_pairs(objs: list[DetectedObject]):
    for a, b in itertools.combinations(objs, 2):
        if a.charge * b.charge < 0:
            yield a, b


def _leading_pair(objs):
    pairs = list(_os_pairs(objs))
    if not pairs:
        return None
    return max(pairs, key=lambda ab: ab[0].pt + ab[1].pt)


def _candidate(fs, objs, det: DetectorEvent, sub=()) -> Candidate:
    return Candidate(fs, list(objs), invariant_mass(objs), det.truth.true_mass, tuple(sub))


def reconstruct(det: DetectorEvent, final_state: str = "auto",
                sel: Selection = Selection()) -> Candidate | None:
    """Build the best candidate of the requested final state (or choose automatically)."""
    mus = det.objects_of(ds.MUON, min_pt=sel.lepton_min_pt)
    els = det.objects_of(ds.ELECTRON, min_pt=sel.lepton_min_pt)
    phs = det.objects_of(ds.PHOTON, min_pt=sel.photon_min_pt)

    if final_state == "mumu":
        pair = _leading_pair(mus)
        return _candidate("mumu", pair, det) if pair else None
    if final_state == "ee":
        pair = _leading_pair(els)
        return _candidate("ee", pair, det) if pair else None
    if final_state == "ll":
        cands = [c for c in (reconstruct(det, "mumu", sel), reconstruct(det, "ee", sel)) if c]
        return max(cands, key=lambda c: sum(o.pt for o in c.objects), default=None)
    if final_state == "emu":
        pairs = [(e, m) for e in els for m in mus if e.charge * m.charge < 0]
        if not pairs:
            return None
        return _candidate("emu", max(pairs, key=lambda p: p[0].pt + p[1].pt), det)
    if final_state == "gammagamma":
        return _candidate("gammagamma", phs[:2], det) if len(phs) >= 2 else None
    if final_state == "4l":
        return _reconstruct_4l(det, mus, els, sel)
    if final_state == "auto":
        for fs in ("4l",):
            c = reconstruct(det, fs, sel)
            if c:
                return c
        n_lep = len(mus) + len(els)
        if len(phs) >= 2 and phs[1].pt >= sel.diphoton_lead_pt and n_lep < 2:
            return reconstruct(det, "gammagamma", sel)
        for fs in ("ll", "emu", "gammagamma"):
            c = reconstruct(det, fs, sel)
            if c:
                return c
        return None
    raise ValueError(f"unknown final state {final_state!r}")


def _reconstruct_4l(det, mus, els, sel: Selection) -> Candidate | None:
    """Two opposite-sign same-flavour pairs; Z₁ = pair closest to m_Z (CMS/ATLAS-style)."""
    sf_pairs = list(_os_pairs(mus)) + list(_os_pairs(els))
    best = None
    for p1, p2 in itertools.combinations(sf_pairs, 2):
        if {id(o) for o in p1} & {id(o) for o in p2}:
            continue
        m1, m2 = invariant_mass(p1), invariant_mass(p2)
        if abs(m2 - sel.z_mass) < abs(m1 - sel.z_mass):
            p1, p2, m1, m2 = p2, p1, m2, m1
        key = (abs(m1 - sel.z_mass), -sum(o.pt for o in (*p1, *p2)))
        if best is None or key < best[0]:
            best = (key, p1 + p2, (m1, m2))
    if best is None:
        return None
    return _candidate("4l", best[1], det, best[2])


#: which final state to reconstruct for each physics_engine channel
FINAL_STATE_FOR_CHANNEL: dict[str, str] = {
    "jpsi_mumu": "mumu", "jpsi_ee": "ee", "z_mumu": "mumu", "z_ee": "ee",
    "dy_mumu": "mumu", "dy_ee": "ee", "h_gammagamma": "gammagamma",
    "diphoton": "gammagamma", "h_zz_4l": "4l", "zz_4l": "4l", "ww_lnulnu": "ll",
}


# --------------------------------------------------------------------------------------
# Histogram & resonance fit
# --------------------------------------------------------------------------------------


@dataclass
class Histogram:
    lo: float
    hi: float
    n_bins: int
    counts: np.ndarray = field(init=False)

    def __post_init__(self):
        self.counts = np.zeros(self.n_bins, dtype=np.int64)

    @property
    def edges(self) -> np.ndarray:
        return np.linspace(self.lo, self.hi, self.n_bins + 1)

    @property
    def centers(self) -> np.ndarray:
        e = self.edges
        return 0.5 * (e[1:] + e[:-1])

    @property
    def bin_width(self) -> float:
        return (self.hi - self.lo) / self.n_bins

    @property
    def errors(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.counts, 1))

    @property
    def entries(self) -> int:
        return int(self.counts.sum())

    def fill(self, values) -> None:
        v = np.atleast_1d(np.asarray(values, dtype=float))
        c, _ = np.histogram(v, bins=self.edges)
        self.counts += c

    def ascii(self, width: int = 60, fit: FitResult | None = None) -> str:
        peak = max(int(self.counts.max()), 1)
        model = fit.expected_counts(self) if fit and fit.success else None
        rows = []
        for i, (c, n) in enumerate(zip(self.centers, self.counts)):
            bar = "#" * int(round(width * n / peak))
            if model is not None:
                pos = int(round(width * model[i] / peak))
                bar = bar.ljust(width + 1)
                bar = bar[:pos] + "*" + bar[pos + 1:]
            rows.append(f"{c:9.2f} |{bar.rstrip()}  {n}")
        return "\n".join(rows)


FIT_MODELS = ("voigt", "breit_wigner", "lorentzian", "gaussian")
_MODEL_PARAMS = {
    "voigt": ("n_sig", "mass", "width", "sigma", "b0", "b1", "b2"),
    "breit_wigner": ("n_sig", "mass", "width", "b0", "b1", "b2"),
    "lorentzian": ("n_sig", "mass", "width", "b0", "b1", "b2"),
    "gaussian": ("n_sig", "mass", "sigma", "b0", "b1", "b2"),
}


def signal_pdf(model: str, m, mass: float, width: float, sigma: float):
    """Normalised peak shapes in M (1/GeV)."""
    m = np.asarray(m, dtype=float)
    if model == "voigt":  # Breit–Wigner (Lorentzian) ⊗ Gaussian detector resolution
        return voigt_profile(m - mass, sigma, 0.5 * width)
    if model == "breit_wigner":
        return pe.relativistic_breit_wigner_pdf(m, mass, width)
    if model == "lorentzian":
        hw = 0.5 * width
        return (hw / math.pi) / ((m - mass) ** 2 + hw * hw)
    if model == "gaussian":
        return np.exp(-0.5 * ((m - mass) / sigma) ** 2) / (math.sqrt(2 * math.pi) * sigma)
    raise ValueError(f"unknown model {model!r}; choose from {FIT_MODELS}")


def background_density(m, b0: float, b1: float, b2: float, lo: float, hi: float):
    """Smooth falling continuum (events/GeV): b0 · exp(b1 t + b2 t²), t = (m−lo)/(hi−lo)."""
    t = (np.asarray(m, dtype=float) - lo) / (hi - lo)
    return b0 * np.exp(np.clip(b1 * t + b2 * t * t, -700, 700))


@dataclass
class FitResult:
    model: str
    success: bool
    message: str
    params: dict[str, float]
    errors: dict[str, float]
    chi2: float = math.nan
    ndf: int = 0
    window: tuple[float, float] = (math.nan, math.nan)
    s_window: float = math.nan
    b_window: float = math.nan
    lo: float = 0.0
    hi: float = 1.0

    # convenience accessors ------------------------------------------------------------
    @property
    def mass(self) -> float:
        return self.params.get("mass", math.nan)

    @property
    def mass_err(self) -> float:
        return self.errors.get("mass", math.nan)

    @property
    def width(self) -> float:
        return self.params.get("width", math.nan)

    @property
    def width_err(self) -> float:
        return self.errors.get("width", math.nan)

    @property
    def sigma(self) -> float:
        return self.params.get("sigma", math.nan)

    @property
    def n_signal(self) -> float:
        return self.params.get("n_sig", math.nan)

    @property
    def significance(self) -> float:
        """Simple local significance S/√B in the peak window."""
        return self.s_window / math.sqrt(self.b_window) if self.b_window > 0 else math.inf

    @property
    def significance_asimov(self) -> float:
        s, b = self.s_window, self.b_window
        if not (s > 0 and b > 0):
            return math.nan
        return math.sqrt(2 * ((s + b) * math.log(1 + s / b) - s))

    def _full(self) -> dict[str, float]:
        p = {"width": 0.0, "sigma": 1.0, "b2": 0.0}
        p.update(self.params)
        return p

    def signal_density(self, m):
        p = self._full()
        return p["n_sig"] * signal_pdf(self.model, m, p["mass"], p["width"], p["sigma"])

    def background_density(self, m):
        p = self._full()
        return background_density(m, p["b0"], p["b1"], p["b2"], self.lo, self.hi)

    def density(self, m):
        return self.signal_density(m) + self.background_density(m)

    def expected_counts(self, h: Histogram) -> np.ndarray:
        return _bin_integral(self.density, h)


def _bin_integral(density, h: Histogram, n_sub: int = 5) -> np.ndarray:
    e = h.edges
    w = h.bin_width
    sub = e[:-1, None] + (np.arange(n_sub)[None, :] + 0.5) * (w / n_sub)
    return density(sub).mean(axis=1) * w


def fit_resonance(
    h: Histogram,
    model: str = "voigt",
    mass_hint: float | None = None,
    width_hint: float | None = None,
    fix_width: float | None = None,
) -> FitResult:
    """χ² fit (``scipy.optimize.curve_fit``) of peak + exponential continuum to ``h``."""
    if model not in FIT_MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {FIT_MODELS}")
    names = list(_MODEL_PARAMS[model])
    if fix_width is not None and "width" in names:
        names.remove("width")
    lo, hi, span = h.lo, h.hi, h.hi - h.lo
    y = h.counts.astype(float)
    x = h.centers
    if h.entries < 5:
        return FitResult(model, False, "too few entries", {}, {}, lo=lo, hi=hi)

    # ---- starting values from the data ----
    if mass_hint is None or not lo < mass_hint < hi:
        smooth = np.convolve(y, np.ones(3) / 3, mode="same")
        mass_hint = float(x[int(np.argmax(smooth))])
    core = np.abs(x - mass_hint) < 0.1 * span
    side = ~core
    if side.sum() >= 3:
        t = (x[side] - lo) / span
        b1, logb0 = np.polyfit(t, np.log(y[side] + 1.0), 1)
        b0 = math.exp(logb0) / h.bin_width
    else:
        b1, b0 = 0.0, max(y.min(), 1.0) / h.bin_width
    bkg_guess = background_density(x, b0, b1, 0.0, lo, hi) * h.bin_width
    n0 = max(float(y.sum() - bkg_guess.sum()), 0.1 * y.sum(), 1.0)
    w0 = width_hint if width_hint else 0.02 * mass_hint
    guess = {"n_sig": n0, "mass": mass_hint, "width": max(w0, 1e-4 * span),
             "sigma": 0.015 * mass_hint, "b0": max(b0, 1e-3), "b1": float(np.clip(b1, -40, 40)),
             "b2": 0.0}
    bounds = {"n_sig": (0.0, 10.0 * y.sum() + 10), "mass": (lo, hi),
              "width": (1e-6 * span, span), "sigma": (1e-4 * span, 0.5 * span),
              "b0": (0.0, np.inf), "b1": (-50.0, 50.0), "b2": (-50.0, 50.0)}

    def unpack(free) -> dict[str, float]:
        p = {"width": fix_width if fix_width is not None else 0.0, "sigma": 1.0, "b2": 0.0}
        p.update(dict(zip(names, free)))
        return p

    def f(_x, *free):
        p = unpack(free)
        dens = lambda m: (p["n_sig"] * signal_pdf(model, m, p["mass"], p["width"], p["sigma"])  # noqa: E731
                          + background_density(m, p["b0"], p["b1"], p["b2"], lo, hi))
        return _bin_integral(dens, h)

    p0 = [float(np.clip(guess[n], *bounds[n])) for n in names]
    lb = [bounds[n][0] for n in names]
    ub = [bounds[n][1] for n in names]
    try:
        popt, pcov = curve_fit(f, x, y, p0=p0, sigma=h.errors, absolute_sigma=True,
                               bounds=(lb, ub), maxfev=40000)
    except (RuntimeError, ValueError) as exc:
        return FitResult(model, False, str(exc), {}, {}, lo=lo, hi=hi)
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None)) if np.all(np.isfinite(pcov)) else \
        np.full(len(popt), np.nan)
    params = dict(zip(names, map(float, popt)))
    if fix_width is not None and "width" in _MODEL_PARAMS[model]:
        params["width"] = fix_width
    res = FitResult(model, True, "ok", params, dict(zip(names, map(float, perr))), lo=lo, hi=hi)
    mu = res.expected_counts(h)
    res.chi2 = float(np.sum((y - mu) ** 2 / h.errors**2))
    res.ndf = int(h.n_bins - len(names))

    # ---- S and B in the peak window M ± 2σ_eff ----
    p = res._full()
    sig_eff = {"voigt": math.hypot(p["sigma"], 0.5 * p["width"]),
               "gaussian": p["sigma"]}.get(model, 0.5 * p["width"])
    a, b = max(lo, res.mass - 2 * sig_eff), min(hi, res.mass + 2 * sig_eff)
    if b > a:
        grid = np.linspace(a, b, 801)
        res.window = (a, b)
        res.s_window = float(np.trapezoid(res.signal_density(grid), grid))
        res.b_window = float(np.trapezoid(res.background_density(grid), grid))
    return res


# --------------------------------------------------------------------------------------
# Luminosity run
# --------------------------------------------------------------------------------------


@dataclass
class EventRecord:
    channel: str
    is_signal: bool
    truth_mass: float
    reco_mass: float | None
    in_window: bool


class LuminosityRun:
    """Accumulating signal + background run — call ``run(n)`` repeatedly, then ``fit()``."""

    def __init__(
        self,
        signal_channel: str = "z_mumu",
        beam: pe.BeamConfig | None = None,
        signal_fraction: float = 0.5,
        mass_window: tuple[float, float] | None = None,
        n_bins: int = 60,
        seed: int | None = None,
        detector_config: DetectorConfig | None = None,
        background_channel: str | None = None,
        selection: Selection = Selection(),
    ):
        ch = pe.CHANNELS[signal_channel]
        if signal_channel not in FINAL_STATE_FOR_CHANNEL or ch.kind != "signal":
            raise ValueError(f"{signal_channel!r} is not a reconstructable signal channel")
        self.signal = ch
        self.background = background_channel or ch.background
        self.window = mass_window or ch.background_window
        if self.window is None:
            raise ValueError(f"no default mass window for {signal_channel!r}")
        self.final_state = FINAL_STATE_FOR_CHANNEL[signal_channel]
        self.signal_fraction = signal_fraction
        self.beam = beam or pe.BeamConfig("pp", 1000.0)
        rng = np.random.default_rng(seed)
        self._rng = rng
        self.generator = pe.CollisionGenerator(self.beam, seed=int(rng.integers(2**63)))
        self.detector = ds.Detector(detector_config, seed=int(rng.integers(2**63)))
        self.selection = selection
        self.histogram = Histogram(self.window[0], self.window[1], n_bins)
        self.records: list[EventRecord] = []
        #: most recent simulated event / the most recent one that entered the histogram,
        #: as (DetectorEvent, Candidate | None) — lets a live display show a representative
        #: event without re-simulating anything.
        self.last_event: tuple[DetectorEvent, Candidate | None] | None = None
        self.last_selected: tuple[DetectorEvent, Candidate] | None = None

    def _one(self) -> EventRecord:
        is_sig = self.background is None or self._rng.uniform() < self.signal_fraction
        if is_sig:
            ev = self.generator.generate(self.signal.key)
        else:
            ev = self.generator.generate(self.background, mass_window=self.window)
        det = self.detector.simulate(ev)
        cand = reconstruct(det, self.final_state, self.selection)
        m = cand.mass if cand else None
        inside = m is not None and self.window[0] <= m < self.window[1]
        if inside:
            self.histogram.fill(m)
            self.last_selected = (det, cand)
        self.last_event = (det, cand)
        rec = EventRecord(ev.channel, is_sig, ev.true_mass, m, inside)
        self.records.append(rec)
        return rec

    def run(self, n: int) -> list[EventRecord]:
        return [self._one() for _ in range(n)]

    @property
    def n_events(self) -> int:
        return len(self.records)

    @property
    def efficiency(self) -> float:
        return sum(r.in_window for r in self.records) / max(len(self.records), 1)

    def fit(self, model: str = "voigt", fix_width: float | None = None) -> FitResult:
        res_spec = pe.pdg(self.signal.resonance)
        return fit_resonance(self.histogram, model, mass_hint=res_spec.mass,
                             width_hint=max(res_spec.width, 1e-3), fix_width=fix_width)

    def truth_counts_in(self, window: tuple[float, float]) -> tuple[int, int]:
        a, b = window
        s = sum(1 for r in self.records if r.in_window and r.is_signal and a <= r.reco_mass < b)
        bk = sum(1 for r in self.records if r.in_window and not r.is_signal
                 and a <= r.reco_mass < b)
        return s, bk


# --------------------------------------------------------------------------------------
# Mystery Event Challenge
# --------------------------------------------------------------------------------------

MYSTERY_POOL: tuple[str, ...] = (
    "z_mumu", "z_ee", "z_tautau", "h_gammagamma", "h_zz_4l", "ww_lnulnu", "jpsi_mumu",
)

#: Step-A answer categories for the game ("which final-state signature is this?")
FINAL_STATE_CATEGORIES: dict[str, str] = {
    "ee": "e⁺e⁻", "mumu": "μ⁺μ⁻", "gammagamma": "γγ", "4l": "4ℓ",
    "tautau": "τ⁺τ⁻", "llnunu": "ℓ⁺ℓ⁻ + νν̄",
}
SIGNATURE_CATEGORY: dict[str, str] = {
    "z_mumu": "mumu", "jpsi_mumu": "mumu", "z_ee": "ee", "z_tautau": "tautau",
    "h_gammagamma": "gammagamma", "h_zz_4l": "4l", "ww_lnulnu": "llnunu",
}
#: (truth, guess) pairs worth half credit, with the lesson they teach
PARTIAL_CATEGORIES: dict[tuple[str, str], str] = {
    ("llnunu", "ee"): "You found the leptons but missed the neutrinos — look at the missing E_T!",
    ("llnunu", "mumu"): "You found the leptons but missed the neutrinos — look at the missing E_T!",
    ("tautau", "llnunu"): "Right that neutrinos escaped — but they came from τ decays: note the "
                          "displaced tracks that miss the collision vertex.",
    ("llnunu", "tautau"): "Right that neutrinos escaped — but these leptons point straight back "
                          "to the collision vertex (prompt W decays), so no τ was involved.",
    ("tautau", "ee"): "The visible electron(s) came from a τ decay — check the displaced tracks "
                      "and the missing E_T.",
    ("tautau", "mumu"): "The visible muon(s) came from a τ decay — check the displaced tracks "
                        "and the missing E_T.",
}
#: why each process looks the way it does in the detector
PHYSICS_WHY: dict[str, str] = {
    "z_mumu": "Muons are minimum-ionising particles: they cross the ECAL (≈0.3 GeV) and HCAL "
              "(≈1.5 GeV) almost untouched and are the only charged particles that reach the outer "
              "muon spectrometer. Two opposite-charge muons with M_inv ≈ 91.19 GeV are a Z⁰.",
    "jpsi_mumu": "Same muon signature as the Z⁰ — but M_inv ≈ 3.097 GeV: the J/ψ (cc̄) bound "
                 "state. Its low-momentum muons curl strongly in the 3.8 T field.",
    "z_ee": "Electrons leave a curved track and then shower completely in the ECAL (a few "
            "radiation lengths stop them), so nothing reaches the HCAL. Two of them with "
            "M_inv ≈ 91 GeV are a Z⁰.",
    "z_tautau": "The τ lepton lives only cτ ≈ 87 µm, but with βγ ≈ 25 it flies a few mm before "
                "decaying — its daughter tracks miss the collision vertex (displaced vertex). Every "
                "τ decay emits one or two neutrinos, so there is missing E_T and the visible mass "
                "sits well below m_Z.",
    "h_gammagamma": "Photons are neutral: no tracker hits and no curvature — just isolated ECAL "
                    "towers. Two such towers with M_inv ≈ 125 GeV are the Higgs diphoton signature "
                    "(the 2012 discovery channel).",
    "h_zz_4l": "Four leptons in two opposite-charge same-flavour pairs: one pair at m_Z (on-shell Z), "
               "the other far below it (off-shell Z*), all four combining to M_inv ≈ 125 GeV — "
               "the 'golden' Higgs channel.",
    "ww_lnulnu": "Each W decays to a charged lepton plus a neutrino. The two neutrinos escape "
                 "unseen, leaving large missing E_T, and M_ℓℓ shows no resonance peak. The leptons "
                 "are prompt — they point straight back to the collision vertex.",
}
DISPLACED_D0_MM = 0.03  # |d₀| above 30 µm ⇒ track does not come from the primary vertex
DISPLACED_MIN_PT = 2.0  # GeV — ignore soft (underlying-event K⁰_S) tracks


def transverse_impact_parameter_mm(track, primary_vertex_mm) -> float:
    """|d₀|: transverse distance of closest approach of a track's helix to the primary vertex.
    Purely detector-level (helix geometry), so it may be shown to the Mystery player."""
    h = track.helix
    px, py = primary_vertex_mm[0] * ds.MM_TO_M, primary_vertex_mm[1] * ds.MM_TO_M
    if h.h == 0:
        d = abs((h.x0 - px) * math.sin(h.phi0) - (h.y0 - py) * math.cos(h.phi0))
    else:
        xc, yc = h.center
        d = abs(math.hypot(xc - px, yc - py) - h.radius)
    return d / ds.MM_TO_M
_RESONANCE_IDS = {pe.Z0, pe.W_PLUS, -pe.W_PLUS, pe.HIGGS, pe.J_PSI}
_SIGNATURE_KIND = {pe.ELECTRON: "electron", pe.MUON: "muon", pe.PHOTON: "photon",
                   pe.NU_E: "neutrino", pe.NU_MU: "neutrino", pe.NU_TAU: "neutrino"}
_ALIASES = {
    "e": "electron", "e+": "electron", "e-": "electron", "electron": "electron",
    "electrons": "electron", "positron": "electron",
    "mu": "muon", "mu+": "muon", "mu-": "muon", "muon": "muon", "muons": "muon", "μ": "muon",
    "gamma": "photon", "photon": "photon", "photons": "photon", "γ": "photon",
    "nu": "neutrino", "neutrino": "neutrino", "neutrinos": "neutrino", "ν": "neutrino",
    "met": "neutrino", "missing energy": "neutrino",
}
PARENT_OF = {"z_mumu": "Z0", "z_ee": "Z0", "z_tautau": "Z0", "h_gammagamma": "H0", "h_zz_4l": "H0",
             "ww_lnulnu": "W+W-", "jpsi_mumu": "J/psi"}
LARGE_MET_GEV = 25.0
HARD_PHOTON_PT = 10.0


def normalize_particle_guess(guess: Mapping[str, int] | Iterable[str]) -> Counter:
    items = guess.items() if isinstance(guess, Mapping) else ((g, 1) for g in guess)
    out: Counter = Counter()
    for name, n in items:
        key = _ALIASES.get(str(name).strip().lower())
        if key is None:
            raise ValueError(f"unknown particle name {name!r}")
        out[key] += int(n)
    return out


def truth_signature(event: pe.Event) -> Counter:
    """Final-state e/μ/γ/ν that descend from the hard resonance(s) (ISR/UE excluded)."""
    sig: Counter = Counter()
    for p in event.final_state():
        root = p
        while root.mother is not None:
            root = event.particles[root.mother]
        if root.pdg_id in _RESONANCE_IDS and root is not p:
            kind = _SIGNATURE_KIND.get(abs(p.pdg_id))
            if kind:
                sig[kind] += 1
    return sig


@dataclass
class Clues:
    """Detector-level facts only — safe to show the player."""

    layers_lit: list[str]
    n_tracks: int
    tracker_hits: int
    ecal_energy: float
    hcal_energy: float
    muon_hits: int
    objects: dict[str, int]  # hard identified objects
    met: float
    met_phi: float
    large_met: bool
    candidate_final_state: str | None
    candidate_mass: float | None
    hints: list[str]  # level-2 "senior physicist" interpretation hints
    displaced_tracks: int = 0  # tracks with pT ≥ 2 GeV and |d₀| > 30 µm
    max_d0_mm: float = 0.0
    layer_hints: list[str] = field(default_factory=list)  # level-1: which layers fired

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def evidence(self) -> dict[str, Any]:
        """Raw detector-level numbers only (no interpretation, no object identification)."""
        d = self.as_dict()
        for k in ("hints", "layer_hints", "objects", "candidate_final_state"):
            d.pop(k)
        return d


@dataclass
class MysteryEvent:
    challenge_id: str
    detector_event: DetectorEvent  # contains truth — never send to the player directly
    clues: Clues
    _truth_channel: str = field(repr=False)
    _truth_signature: Counter = field(repr=False)

    def public_view(self) -> dict[str, Any]:
        view = self.detector_event.to_dict()
        for t in view["traces"]:  # strip truth identities from the rendering payload
            t.pop("pdg_id", None)
            t.pop("name", None)
            t.pop("class", None)
            t.pop("fate", None)
        return {"challenge_id": self.challenge_id, "clues": self.clues.as_dict(),
                "display": view}


@dataclass
class Grade:
    particles_correct: bool
    particles_partial: bool
    channel_correct: bool
    parent_correct: bool
    score: int
    truth_channel: str
    truth_label: str
    truth_signature: dict[str, int]
    explanation: list[str]
    max_score: int = 100
    truth_category: str | None = None  # Step-A category of the truth (e.g. "tautau")
    guess_category: str | None = None  # set when the guess was given as a category
    lesson: str = ""  # why the signature looks the way it does (PHYSICS_WHY)


class MysteryChallenge:
    def __init__(self, beam: pe.BeamConfig | None = None, seed: int | None = None,
                 detector_config: DetectorConfig | None = None,
                 pool: Iterable[str] = MYSTERY_POOL, selection: Selection = Selection()):
        self.beam = beam or pe.BeamConfig("pp", 1000.0)
        self.rng = np.random.default_rng(seed)
        self.generator = pe.CollisionGenerator(self.beam, seed=int(self.rng.integers(2**63)))
        self.detector = ds.Detector(detector_config, seed=int(self.rng.integers(2**63)))
        self.pool = tuple(pool)
        self.selection = selection
        self.events: dict[str, MysteryEvent] = {}

    # ---- event creation --------------------------------------------------------------
    def new_event(self, channel: str | None = None, max_tries: int = 100) -> MysteryEvent:
        channel = channel or self.pool[int(self.rng.integers(len(self.pool)))]
        det = None
        for _ in range(max_tries):  # keep the puzzle solvable: signature must be visible
            ev = self.generator.generate(channel)
            det = self.detector.simulate(ev)
            if self._signature_visible(det):
                break
        assert det is not None
        me = MysteryEvent(uuid.uuid4().hex[:12], det, self.make_clues(det),
                          channel, truth_signature(det.truth))
        self.events[me.challenge_id] = me
        return me

    def _signature_visible(self, det: DetectorEvent) -> bool:
        sig = truth_signature(det.truth)
        seen: Counter = Counter()
        for o in det.objects:
            p = det.truth.particles[o.truth_index]
            kind = _SIGNATURE_KIND.get(abs(p.pdg_id))
            if kind and kind == o.kind:
                seen[kind] += 1
        visible = {k: n for k, n in sig.items() if k != "neutrino"}
        return all(seen[k] >= n for k, n in visible.items())

    def make_clues(self, det: DetectorEvent) -> Clues:
        sel = self.selection
        s = det.layer_summary()
        objs = {
            "muon": len(det.objects_of(ds.MUON, min_pt=sel.lepton_min_pt)),
            "electron": len(det.objects_of(ds.ELECTRON, min_pt=sel.lepton_min_pt)),
            "photon": len(det.objects_of(ds.PHOTON, min_pt=HARD_PHOTON_PT)),
            "charged_hadron": len(det.objects_of(ds.CHARGED_HADRON)),
            "neutral_hadron": len(det.objects_of(ds.NEUTRAL_HADRON)),
        }
        layers = ["tracker"] if s["n_tracks"] else []
        if s["ecal_energy"] > 0.5:
            layers.append("ecal")
        if s["hcal_energy"] > 0.5:
            layers.append("hcal")
        if s["muon_hits"]:
            layers.append("muon")
        cand = reconstruct(det, "auto", sel)
        large = det.met > LARGE_MET_GEV
        pv = det.truth.primary_vertex
        d0s = [transverse_impact_parameter_mm(t.track, pv) for t in det.traces
               if t.track and t.track.reconstructed and t.track.pt_measured >= DISPLACED_MIN_PT]
        displaced = sum(d > DISPLACED_D0_MM for d in d0s)
        max_d0 = max(d0s, default=0.0)

        ecal_towers = det.tower_energies(Layer.ECAL)
        hcal_towers = det.tower_energies(Layer.HCAL)
        n_mu_tracks = sum(1 for t in det.traces if t.track and t.track.muon_hits)
        layer_hints = [
            f"Silicon tracker: {s['n_tracks']} reconstructed tracks with {s['tracker_hits']} hits.",
            f"ECAL: {s['ecal_energy']:.1f} GeV in {int((ecal_towers > 0.5).sum())} towers above "
            f"0.5 GeV (hottest tower {ecal_towers.max():.1f} GeV).",
            f"HCAL: {s['hcal_energy']:.1f} GeV in {int((hcal_towers > 0.5).sum())} towers above "
            f"0.5 GeV (hottest tower {hcal_towers.max():.1f} GeV).",
            f"Muon chambers: {s['muon_hits']} hits" + (f" on {n_mu_tracks} track(s)."
                                                       if n_mu_tracks else " — silent."),
            f"Vertexing: {displaced} hard track(s) with |d₀| > {DISPLACED_D0_MM * 1e3:.0f} µm "
            f"(largest |d₀| = {max_d0 * 1e3:.0f} µm).",
            f"Missing transverse energy: {det.met:.1f} GeV.",
        ]

        hints = []
        if objs["muon"]:
            hints.append(f"{objs['muon']} track(s) punch through all calorimeters into the "
                         "muon chambers — the signature of muons.")
        if objs["electron"]:
            hints.append(f"{objs['electron']} track(s) end in a compact ECAL shower — "
                         "electrons.")
        if objs["photon"]:
            hints.append(f"{objs['photon']} energetic ECAL shower(s) with NO track — photons.")
        hints.append(f"Missing transverse energy is {det.met:.1f} GeV"
                     + (" — something invisible escaped (neutrinos?)." if large
                        else " — consistent with zero (only resolution effects)."))
        if displaced:
            hints.append(f"{displaced} energetic track(s) miss the collision vertex by more than "
                         f"{DISPLACED_D0_MM * 1e3:.0f} µm — a short-lived particle decayed in flight "
                         "(cτ of order 100 µm).")
        if cand:
            hints.append(f"The best {cand.label()} candidate has M_inv = {cand.mass:.2f} GeV.")
        return Clues(layers, s["n_tracks"], s["tracker_hits"], s["ecal_energy"],
                     s["hcal_energy"], s["muon_hits"], objs, det.met, det.met_phi, large,
                     cand.final_state if cand else None, cand.mass if cand else None, hints,
                     displaced, max_d0, layer_hints)

    # ---- grading ---------------------------------------------------------------------
    def grade(self, mystery: MysteryEvent | str,
              particles: str | Mapping[str, int] | Iterable[str], channel: str) -> Grade:
        """Score a claim (100 pts: 50 final state + 50 channel, with partial credit).

        ``particles`` is either a Step-A category key (``FINAL_STATE_CATEGORIES``, e.g.
        ``"tautau"``) or explicit particle counts (``{"muon": 2}`` / ``["mu+", "mu-"]``).
        """
        me = self.events[mystery] if isinstance(mystery, str) else mystery
        truth = me._truth_signature
        truth_cat = SIGNATURE_CATEGORY.get(me._truth_channel)
        guess_cat = None
        lesson_partial = None
        if isinstance(particles, str):
            guess_cat = particles.strip()
            if guess_cat not in FINAL_STATE_CATEGORIES:
                raise ValueError(f"unknown final-state category {particles!r}; "
                                 f"choose from {sorted(FINAL_STATE_CATEGORIES)}")
            exact = guess_cat == truth_cat
            lesson_partial = PARTIAL_CATEGORIES.get((truth_cat, guess_cat))
            partial = not exact and lesson_partial is not None
        else:
            guess = normalize_particle_guess(particles)
            exact = +guess == +truth
            visible_truth = Counter({k: v for k, v in truth.items() if k != "neutrino"})
            visible_guess = Counter({k: v for k, v in guess.items() if k != "neutrino"})
            partial = not exact and visible_guess == visible_truth
            if partial:
                lesson_partial = ("You found every visible particle but missed the invisible "
                                  "neutrinos — look at the missing transverse energy!")
        ch_ok = channel == me._truth_channel
        parent_ok = PARENT_OF.get(channel) == PARENT_OF.get(me._truth_channel)
        score = (50 if exact else 25 if partial else 0) + (50 if ch_ok else 25 if parent_ok else 0)

        det = me.detector_event
        ev = det.truth
        label = pe.CHANNELS[me._truth_channel].label
        expl = [f"Truth: {label}  (generated M = {ev.true_mass:.3f} GeV)."]
        sig_txt = ", ".join(f"{n}× {k}" for k, n in sorted(truth.items())) or "hadrons only"
        expl.append(f"Signature particles from the hard decay: {sig_txt}.")
        if lesson_partial:
            expl.append(lesson_partial)
        if not ch_ok and parent_ok:
            expl.append("Right parent particle, wrong decay mode.")
        if me.clues.candidate_mass is not None:
            parent = pe.CHANNELS[me._truth_channel].resonance
            expl.append(f"Reconstructed M_inv = {me.clues.candidate_mass:.2f} GeV vs PDG "
                        f"m({PARENT_OF[me._truth_channel]}) = {pe.pdg(parent).mass:.3f} GeV.")
        mex, mey, nu_pt = ev.truth_missing_pt()
        if ev.neutrinos():
            expl.append(f"{len(ev.neutrinos())} neutrino(s) escaped with Σp_T = {nu_pt:.1f} GeV; "
                        f"the detector measured missing E_T = {det.met:.1f} GeV.")
        flights = [p for p in ev.particles if abs(p.pdg_id) == pe.TAU and p.decay_vertex]
        if flights:
            expl.append("τ flight lengths: " + ", ".join(f"{p.flight_length_mm:.2f} mm"
                                                         for p in flights)
                        + f" (βγcτ, cτ = {pe.pdg(pe.TAU).ctau_mm * 1e3:.0f} µm); "
                        f"{me.clues.displaced_tracks} displaced track(s) seen.")
        return Grade(exact, partial, ch_ok, parent_ok, score, me._truth_channel, label,
                     dict(truth), expl, truth_category=truth_cat, guess_category=guess_cat,
                     lesson=PHYSICS_WHY.get(me._truth_channel, ""))


def deduce_from_clues(clues: Clues) -> tuple[str, str]:
    """A rule-based 'junior physicist' that answers using ONLY the public clues.
    Returns (final-state category, channel). Demonstrates (and tests) that every mystery is
    solvable without the MC truth."""
    o, m = clues.objects, clues.candidate_mass or 0.0
    fs = clues.candidate_final_state
    n_lep = o["muon"] + o["electron"]
    mz, mh, mj = pe.pdg(pe.Z0).mass, pe.pdg(pe.HIGGS).mass, pe.pdg(pe.J_PSI).mass
    if fs == "4l":
        return "4l", "h_zz_4l"
    if (fs == "gammagamma" or o["photon"] >= 2) and abs(m - mh) < 15 and n_lep < 2:
        return "gammagamma", "h_gammagamma"
    if clues.displaced_tracks >= 1:  # tracks missing the vertex: τ decays in flight
        return "tautau", "z_tautau"
    if clues.large_met and n_lep >= 2:
        return "llnunu", "ww_lnulnu"
    if fs in ("mumu", "ee"):
        if fs == "mumu" and abs(m - mj) < 0.3:
            return "mumu", "jpsi_mumu"
        if abs(m - mz) < 12:
            return fs, f"z_{fs}"
    if fs in ("mumu", "ee", "emu") and n_lep >= 2:
        # off-peak lepton pair: prompt W leptons are hard (M_ℓℓ large); τ daughters are soft
        return ("llnunu", "ww_lnulnu") if m > 60 else ("tautau", "z_tautau")
    return "tautau", "z_tautau"


# --------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------


def _resolution_study(beam, channel, n, seed):
    gen = pe.CollisionGenerator(beam, seed=seed)
    det = ds.Detector(seed=seed + 1)
    res = []
    for _ in range(n):
        d = det.simulate(gen.generate(channel))
        c = reconstruct(d, FINAL_STATE_FOR_CHANNEL[channel])
        if c is not None:
            res.append(c.residual)
    r = np.array(res)
    return len(r) / n, float(np.median(r)), float(1.4826 * np.median(np.abs(r - np.median(r))))


def run_self_test(seed: int = 2026) -> bool:
    ok = True
    print("=" * 96)
    print("POCKET CERN COLLIDER - invariance_lab self-test")
    print("=" * 96)

    # 1. single-event reconstruction
    print("\n[1] Single-event reconstruction: smeared M_inv vs Monte-Carlo truth")
    det = ds.Detector(seed=seed)
    for beam, ch in [(pe.BeamConfig("ee", 91.19), "z_mumu"),
                     (pe.BeamConfig("pp", 1000.0), "z_ee"),
                     (pe.BeamConfig("pp", 1000.0), "h_gammagamma"),
                     (pe.BeamConfig("pp", 1000.0), "h_zz_4l"),
                     (pe.BeamConfig("ee", 3.1), "jpsi_mumu")]:
        gen = pe.CollisionGenerator(beam, seed=seed)
        for _ in range(50):
            d = det.simulate(gen.generate(ch))
            c = reconstruct(d, FINAL_STATE_FOR_CHANNEL[ch])
            if c:
                break
        extra = f"  (m_Z1={c.sub_masses[0]:.2f}, m_Z2={c.sub_masses[1]:.2f})" if c.sub_masses else ""
        print(f"  {ch:<14}{beam.label():<26} {c.label():<6} M_inv = {c.mass:9.4f} GeV   "
              f"truth = {c.truth_mass:9.4f} GeV   Δ = {c.residual:+.4f}{extra}")

    print("\n    Resolution study (300 events each): efficiency, median Δ, robust σ(Δ)")
    for beam, ch in [(pe.BeamConfig("pp", 1000.0), "z_mumu"),
                     (pe.BeamConfig("pp", 1000.0), "z_ee"),
                     (pe.BeamConfig("pp", 1000.0), "h_gammagamma"),
                     (pe.BeamConfig("pp", 1000.0), "h_zz_4l"),
                     (pe.BeamConfig("ee", 3.1), "jpsi_mumu")]:
        eff, med, sig = _resolution_study(beam, ch, 300, seed)
        good = abs(med) < 0.01 * pe.pdg(pe.CHANNELS[ch].resonance).mass
        ok &= good
        print(f"    {ch:<14}{beam.label():<26} eff={eff:5.2f}  median Δ={med:+8.4f} GeV  "
              f"σ={sig:7.4f} GeV  {'PASS' if good else 'FAIL'}")

    # 2. luminosity runs
    print("\n[2] Luminosity runs: Voigt (BW ⊗ Gauss) + exponential continuum fits")
    runs = [("z_mumu", 500, 0.5, None), ("z_ee", 2000, 0.5, None),
            ("h_gammagamma", 4000, 0.25, None), ("h_zz_4l", 1000, 0.5, None)]
    z_run = None
    for ch, n, f, win in runs:
        run = LuminosityRun(ch, pe.BeamConfig("pp", 1000.0), signal_fraction=f,
                            mass_window=win, seed=seed + n)
        run.run(n)
        fit = run.fit("voigt")
        m0 = pe.pdg(run.signal.resonance).mass
        s_true, b_true = run.truth_counts_in(fit.window)
        pull = (fit.mass - m0) / fit.mass_err if fit.mass_err > 0 else math.nan
        good = fit.success and abs(fit.mass - m0) < max(4 * fit.mass_err, 0.005 * m0)
        ok &= good
        print(f"  {ch:<13} N={n:<5} sel.eff={run.efficiency:4.2f}  "
              f"M_fit = {fit.mass:8.3f} ± {fit.mass_err:5.3f} GeV (PDG {m0:.3f}, pull {pull:+.1f})"
              f"  Γ_fit = {fit.width:6.3f} ± {fit.width_err:5.3f}  σ_res = {fit.sigma:5.3f}")
        print(f"  {'':<13} χ²/ndf = {fit.chi2:6.1f}/{fit.ndf}   window [{fit.window[0]:.2f}, "
              f"{fit.window[1]:.2f}]  S={fit.s_window:6.1f} B={fit.b_window:6.1f} "
              f"→ S/√B = {fit.significance:5.1f}  (truth S={s_true}, B={b_true}; "
              f"Asimov Z = {fit.significance_asimov:4.1f})  {'PASS' if good else 'FAIL'}")
        if ch == "z_mumu":
            z_run, z_fit = run, fit
    if z_run is not None:
        print("\n  Z → μ⁺μ⁻ + Drell–Yan, 500 events  (# data, * fit)")
        print(z_run.histogram.ascii(width=50, fit=z_fit))

    # 3. Mystery challenge
    print("\n[3] Mystery Event Challenge — rule-based solver uses public clues only")
    game = MysteryChallenge(seed=seed)
    total, n_rounds = 0, 28
    for i in range(n_rounds):
        me = game.new_event(MYSTERY_POOL[i % len(MYSTERY_POOL)])
        guess_fs, guess_ch = deduce_from_clues(me.clues)
        g = game.grade(me, guess_fs, guess_ch)
        total += g.score
        if i < len(MYSTERY_POOL):
            c = me.clues
            m_txt = f"{c.candidate_mass:7.2f}" if c.candidate_mass else "   -   "
            print(f"  layers={'+'.join(c.layers_lit):<24} obj(mu,e,γ)=({c.objects['muon']},"
                  f"{c.objects['electron']},{c.objects['photon']}) MET={c.met:6.1f} "
                  f"M_inv={m_txt}  guess={guess_ch:<13} truth={g.truth_channel:<13} "
                  f"score={g.score}")
    avg = total / n_rounds
    print(f"  average solver score over {n_rounds} rounds: {avg:.1f} / 100")
    ok &= avg >= 80

    print("\n" + ("ALL CHECKS PASSED" if ok else "SELF-TEST FAILED"))
    return ok


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(0 if run_self_test() else 1)
