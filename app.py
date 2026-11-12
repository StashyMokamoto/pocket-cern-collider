"""
app.py — Pocket CERN Collider · FastAPI "Control Room" server
=============================================================

Serves the single-page control room (``static/index.html``) and a small JSON API on top of
the three physics modules:

    GET  /                      → static/index.html
    GET  /api/health            → liveness probe
    GET  /api/config            → beams, channels (PDG masses, thresholds), detector geometry
    POST /api/collide           → one collision: truth → detector → reconstruction payload
    POST /api/luminosity/step   → advance a signal+background run by N events, histogram + fit
    POST /api/mystery/new       → a Mystery Event (MC truth & object IDs stripped)
    POST /api/mystery/hint      → physics hints (level 1: layers, level 2: interpretation)
    POST /api/mystery/guess     → grade a discovery claim, reveal the truth, update stats
    GET  /api/mystery/stats     → a player's running score & streak

Run it:   .venv/Scripts/python app.py          (→ http://127.0.0.1:8000)
     or   .venv/Scripts/python -m uvicorn app:app --port 8000

Units in every payload: GeV for energies/momenta/masses, **metres** for detector-display
coordinates (paths, hits, deposits), **millimetres** for vertex positions (``*_mm`` keys).
Non-finite numbers are sent as ``null``.
"""

from __future__ import annotations

import math
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import detector_sim as ds
import invariance_lab as lab
import physics_engine as pe

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

B_FIELD_RANGE = (0.0, 6.0)  # T
ASYMMETRY_MAX = 0.9  # |β_z| of the CM frame
MAX_EVENTS_PER_STEP = 1000
MIN_ENTRIES_FOR_FIT = 30
MAX_LUMI_RUNS = 8
MAX_MYSTERY_EVENTS = 256
VERTEX_MIN_FLIGHT_MM = 1e-3  # secondary vertices shown in the zoom inset (≥ 1 µm)
FIT_CURVE_POINTS = 240
HINT_COSTS = {1: 5, 2: 15}  # points deducted for physics hint level 1 (layers) / 2 (interpretation)
MAX_PLAYERS = 1000

BeamKind = Literal["ee", "pp"]
FitModel = Literal["voigt", "breit_wigner", "lorentzian", "gaussian"]

# --------------------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------------------


def _f(x: Any, nd: int = 6) -> float | None:
    """Round a scalar for JSON; non-finite → None."""
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _clean(obj: Any) -> Any:
    """Recursively make a (small) structure JSON-safe: numpy → python, NaN/inf → None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _f(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def _xyz(points, nd: int = 4) -> list[list[float]]:
    """(N, 3) metre coordinates → rounded nested lists (always finite in practice)."""
    a = np.round(np.asarray(points, dtype=float).reshape(-1, 3), nd)
    a[~np.isfinite(a)] = 0.0
    return a.tolist()


# --------------------------------------------------------------------------------------
# Beam / channel metadata
# --------------------------------------------------------------------------------------


def make_beam(kind: str, sqrt_s: float, asymmetry: float = 0.0) -> pe.BeamConfig:
    """``asymmetry`` is the velocity β_z of the CM frame along +z (0 = symmetric)."""
    try:
        return pe.BeamConfig(kind=kind, sqrt_s=float(sqrt_s), beta_z=float(asymmetry))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def beam_energies(beam: pe.BeamConfig) -> tuple[float, float]:
    b1, b2 = beam.lab_beams()
    return b1.E, b2.E


def beam_info(beam: pe.BeamConfig) -> dict[str, Any]:
    e1, e2 = beam_energies(beam)
    return {"kind": beam.kind, "sqrt_s": _f(beam.sqrt_s), "beta_z": _f(beam.beta_z),
            "e_beam1": _f(e1), "e_beam2": _f(e2), "label": beam.label()}


def hard_threshold_gev(ch: pe.Channel) -> float:
    """Smallest hard-system mass at which the channel makes physical sense."""
    if ch.key == "ww_lnulnu":
        return 2.0 * pe.pdg(pe.W_PLUS).mass  # nominal on-shell W pair threshold
    if ch.resonance is not None:
        r = pe.pdg(ch.resonance)
        return max(r.mass - 5.0 * r.width, 0.0)  # generator requires the peak in reach
    lep = {"dy_mumu": pe.MUON, "dy_ee": pe.ELECTRON, "dy_tautau": pe.TAU}.get(ch.key)
    if lep is not None:
        return 2.0 * pe.pdg(lep).mass
    if ch.key == "zz_4l":
        return 4.0 * pe.pdg(pe.ELECTRON).mass
    return 0.0


def min_sqrt_s(ch: pe.Channel, kind: str) -> float:
    thr = hard_threshold_gev(ch)
    if kind == "pp":
        thr += 2.0 * pe.pdg(pe.PROTON).mass + 0.1  # two proton remnants must survive
    return max(thr, pe.SQRT_S_MIN)


def channel_info(ch: pe.Channel) -> dict[str, Any]:
    res = None
    if ch.resonance is not None:
        r = pe.pdg(ch.resonance)
        res = {"pdg_id": r.pdg_id, "name": r.name, "mass": _f(r.mass, 6),
               "width": _f(r.width, 9), "ctau_mm": _f(r.ctau_mm, 12)}
    return {
        "key": ch.key, "label": ch.label, "kind": ch.kind, "description": ch.description,
        "resonance": res,
        "threshold_gev": _f(hard_threshold_gev(ch)),
        "min_sqrt_s": {k: _f(min_sqrt_s(ch, k)) for k in ("ee", "pp")},
        "recommended_sqrt_s": dict(ch.recommended_sqrt_s),
        "final_state": lab.FINAL_STATE_FOR_CHANNEL.get(ch.key),
        "background": ch.background,
        "mass_window": list(ch.background_window) if ch.background_window else None,
        "luminosity": ch.kind == "signal" and ch.background_window is not None
                      and ch.key in lab.FINAL_STATE_FOR_CHANNEL,
        "mystery": ch.key in lab.MYSTERY_POOL,
    }


def detector_info(cfg: ds.DetectorConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ds.DetectorConfig()
    labels = {ds.Layer.BEAM_PIPE: "Beam Pipe", ds.Layer.TRACKER: "Silicon Tracker",
              ds.Layer.ECAL: "ECAL", ds.Layer.HCAL: "HCAL", ds.Layer.MUON: "Muon Chambers"}
    return {
        "units": "m",
        "layers": [{"id": layer.value, "label": labels[layer], "r_min": lo, "r_max": hi}
                   for layer, (lo, hi) in ds.LAYER_RADII_M.items()],
        "silicon_layers_m": list(ds.SILICON_LAYERS_M),
        "pixel_layers_m": list(ds.PIXEL_LAYERS_M),
        "muon_stations_m": list(ds.MUON_STATIONS_M),
        "eta_max": cfg.eta_max,
        "ecal_phi_towers": cfg.ecal_phi_towers,
        "hcal_phi_towers": cfg.hcal_phi_towers,
        "tower_deta": cfg.tower_deta,
        "b_field_default": cfg.b_field_t,
        "b_field_range": list(B_FIELD_RANGE),
        "larmor_constant": ds.LARMOR_CONSTANT,
        "resolution": {
            "tracker": f"σ(pT)/pT = {cfg.tracker_a} ⊕ {cfg.tracker_b}·pT",
            "ecal": f"σ(E)/E = {cfg.ecal_stochastic}/√E ⊕ {cfg.ecal_constant}",
            "hcal": f"σ(E)/E = {cfg.hcal_stochastic}/√E ⊕ {cfg.hcal_constant}",
        },
    }


def config_payload() -> dict[str, Any]:
    return {
        "app": "Pocket CERN Collider",
        "units": {"energy": "GeV", "display_length": "m", "vertex_length": "mm", "field": "T"},
        "beams": [
            {"id": "ee", "label": "e⁺e⁻", "particles": ["e-", "e+"], "default_sqrt_s": 91.19},
            {"id": "pp", "label": "pp", "particles": ["p", "p"], "default_sqrt_s": 1000.0},
        ],
        "sqrt_s_range": [pe.SQRT_S_MIN, pe.SQRT_S_MAX],
        "asymmetry_range": [-ASYMMETRY_MAX, ASYMMETRY_MAX],
        "default_sqrt_s": {"ee": 91.19, "pp": 1000.0},
        "channels": [channel_info(ch) for ch in pe.CHANNELS.values()],
        "fit_models": list(lab.FIT_MODELS),
        "detector": detector_info(),
        "mystery": mystery_meta(),
        "limits": {"max_events_per_step": MAX_EVENTS_PER_STEP,
                   "min_entries_for_fit": MIN_ENTRIES_FOR_FIT},
    }


# --------------------------------------------------------------------------------------
# Event payload (truth → detector → reconstruction, for the event display)
# --------------------------------------------------------------------------------------

_event_counter = 0
_counter_lock = threading.Lock()


def _next_event_number() -> int:
    global _event_counter
    with _counter_lock:
        _event_counter += 1
        return _event_counter


def particle_origins(ev: pe.Event) -> list[str]:
    """Per particle: 'hard' (the physics process), 'isr', 'underlying' (pp soft hadrons)
    or 'remnant'. Underlying-event hadrons are created before the beam remnants."""
    first_remnant = next((p.index for p in ev.particles if p.status is pe.Status.REMNANT), None)
    out = []
    for p in ev.particles:
        root = p
        while root.mother is not None:
            root = ev.particles[root.mother]
        if root.status is pe.Status.REMNANT:
            out.append("remnant")
        elif "ISR" in root.name:
            out.append("isr")
        elif first_remnant is not None and root.index < first_remnant:
            out.append("underlying")
        else:
            out.append("hard")
    return out


def event_payload(det: ds.DetectorEvent, cand: lab.Candidate | None, *,
                  hide_truth: bool = False) -> dict[str, Any]:
    """Everything the event display needs.  ``hide_truth`` strips every Monte-Carlo label
    (names, PDG ids, fates, invisible particles, generator masses) — Mystery mode."""
    ev = det.truth
    origins = particle_origins(ev)
    obj_index = {id(o): i for i, o in enumerate(det.objects)}
    obj_by_truth = {o.truth_index: i for i, o in enumerate(det.objects)}
    m_to_mm = 1.0 / ds.MM_TO_M

    traces = []
    for t in det.traces:
        p = ev.particles[t.truth_index]
        invisible = t.particle_class in (ds.NEUTRINO, ds.REMNANT)
        if hide_truth and (invisible or not t.visible_track):
            continue  # a real display shows neither neutrinos nor (truth) photon lines
        pmag = p.p4.p
        u = (p.p4.p3 / pmag).tolist() if pmag > 0 else [0.0, 0.0, 0.0]
        track = t.track
        item: dict[str, Any] = {
            "id": t.truth_index,
            "path": _xyz(t.path),
            "hits": _xyz(track.hits) if track else [],
            "muon_hits": _xyz(track.muon_hits) if track else [],
            "visible_track": t.visible_track,
            "charged": p.charge != 0,
            "object": obj_by_truth.get(t.truth_index),
            "vertex_mm": [_f(c, 6) for c in p.production_vertex],
            "dir": [_f(c, 6) for c in u],
            "radius_m": _f(track.radius_m, 4) if track else None,
            "reconstructed": bool(track.reconstructed) if track else False,
            "d0_mm": _f(lab.transverse_impact_parameter_mm(track, ev.primary_vertex), 5)
                     if track else None,
        }
        if not hide_truth:
            item.update({
                "name": p.name, "pdg_id": p.pdg_id, "class": t.particle_class, "fate": t.fate,
                "origin": origins[p.index], "charge": _f(p.charge, 3),
                "pt": _f(p.p4.pt), "eta": _f(p.p4.eta, 4), "phi": _f(p.p4.phi, 4),
                "E": _f(p.p4.E), "layers": [layer.value for layer in t.layers_reached],
            })
        traces.append(item)

    deposits = [{"layer": d.layer.value, "E": _f(d.energy, 4), "phi": _f(d.phi, 4),
                 "eta": _f(d.eta, 4), "iphi": d.iphi, "ieta": d.ieta,
                 "pos": [_f(c, 4) for c in d.position_m], "id": d.truth_index}
                for d in det.deposits]

    objects = []
    for i, o in enumerate(det.objects):
        item = {
            "index": i, "kind": o.kind, "charge": o.charge,
            "pt": _f(o.pt), "eta": _f(o.eta, 4), "phi": _f(o.phi, 4), "E": _f(o.energy),
            "px": _f(o.p4.px), "py": _f(o.p4.py), "pz": _f(o.p4.pz), "mass": _f(o.p4.mass),
            "id": o.truth_index,
            "n_hits": len(o.track.hits) if o.track else 0,
            "n_muon_hits": len(o.track.muon_hits) if o.track else 0,
            "track_radius_m": _f(o.track.radius_m, 4) if o.track else None,
            "calo_E": _f(sum(d.energy for d in o.deposits), 4),
            "ecal_E": _f(sum(d.energy for d in o.deposits if d.layer is ds.Layer.ECAL), 4),
            "hcal_E": _f(sum(d.energy for d in o.deposits if d.layer is ds.Layer.HCAL), 4),
            "has_track": o.track is not None,
            "d0_mm": _f(lab.transverse_impact_parameter_mm(o.track, ev.primary_vertex), 5)
                     if o.track else None,
        }
        if hide_truth:
            item["kind"] = None  # identification is the player's job in Mystery mode
        else:
            p = ev.particles[o.truth_index]
            item.update({"truth_name": p.name, "truth_pt": _f(p.p4.pt), "truth_E": _f(p.p4.E),
                         "truth_charge": _f(p.charge, 3), "origin": origins[p.index]})
        objects.append(item)

    vertices = []
    for p in ev.particles:
        if p.decay_vertex is None or p.flight_length_mm < VERTEX_MIN_FLIGHT_MM:
            continue
        v = {"production_mm": [_f(c, 6) for c in p.production_vertex],
             "decay_mm": [_f(c, 6) for c in p.decay_vertex],
             "flight_mm": _f(p.flight_length_mm, 6),
             "transverse_mm": _f(math.hypot(p.decay_vertex[0] - p.production_vertex[0],
                                            p.decay_vertex[1] - p.production_vertex[1]), 6),
             "daughters": list(p.daughters)}
        if not hide_truth:
            v.update({"index": p.index, "name": p.name, "pdg_id": p.pdg_id,
                      "charged": p.charge != 0, "betagamma": _f(p.p4.p / p.mass, 4),
                      "ctau_mm": _f(p.ctau_mm, 6), "origin": origins[p.index]})
        vertices.append(v)

    s_px = math.fsum(o.p4.px for o in det.objects)
    s_py = math.fsum(o.p4.py for o in det.objects)
    balance: dict[str, Any] = {
        "sum_px": _f(s_px), "sum_py": _f(s_py), "met_x": _f(det.met_x), "met_y": _f(det.met_y),
        "closure": _f(math.hypot(s_px + det.met_x, s_py + det.met_y), 12),
        "visible_E": _f(math.fsum(o.energy for o in det.objects)),
        "initial_E": _f(ev.initial_p4.E),
    }
    met: dict[str, Any] = {"met": _f(det.met), "phi": _f(det.met_phi, 5),
                           "mex": _f(det.met_x), "mey": _f(det.met_y)}

    candidate = None
    if cand is not None:
        candidate = {"final_state": cand.final_state, "label": cand.label(),
                     "mass": _f(cand.mass), "sub_masses": [_f(m) for m in cand.sub_masses],
                     "objects": [obj_index[id(o)] for o in cand.objects if id(o) in obj_index],
                     "pt": _f(cand.p4.pt), "rapidity": _f(cand.p4.rapidity, 4)}

    payload: dict[str, Any] = {
        "event_number": _next_event_number(),
        "beam": beam_info(ev.beam),
        "b_field_t": det.config.b_field_t,
        "primary_vertex_mm": [_f(c, 6) for c in ev.primary_vertex],
        "traces": traces, "deposits": deposits,
        "towers": {"ecal": [_f(e, 4) for e in det.tower_energies(ds.Layer.ECAL)],
                   "hcal": [_f(e, 4) for e in det.tower_energies(ds.Layer.HCAL)]},
        "objects": objects, "met": met, "candidate": candidate, "vertices": vertices,
        "balance": balance, "summary": _clean(det.layer_summary()),
    }
    if hide_truth:
        payload["summary"].pop("n_invisible", None)
        payload["summary"].pop("n_escaped_forward", None)
        payload["vertices"] = []  # truth decay vertices: not measurable for 1-prong decays
        if candidate is not None:  # keep the mass, not the (identifying) final-state label
            n = len(candidate["objects"])
            candidate.update({"final_state": None, "label": f"{n}-object combination"})
        return payload

    mex, mey, tmet = ev.truth_missing_pt()
    met.update({"truth_met": _f(tmet), "truth_phi": _f(math.atan2(mey, mex), 5)})
    if candidate is not None:
        candidate.update({"truth_mass": _f(cand.truth_mass), "residual": _f(cand.residual)})
    balance.update({"truth_max_dP": _f(ev.max_conservation_violation(), 18),
                    "truth_dP": [_f(c, 18) for c in ev.conservation_residual()],
                    "truth_charge_residual": _f(ev.charge_residual(), 6)})
    ch = pe.CHANNELS[ev.channel]
    payload["truth"] = {
        "channel": ev.channel, "label": ch.label, "kind": ch.kind,
        "is_signal": ev.is_signal, "true_mass": _f(ev.true_mass),
        "n_particles": len(ev.particles), "n_final": len(ev.final_state()),
        "n_underlying": sum(1 for p in ev.final_state() if origins[p.index] == "underlying"),
        "decay_chain": [
            {"index": p.index, "name": p.name, "pdg_id": p.pdg_id, "mother": p.mother,
             "status": p.status.value, "mass": _f(p.mass), "E": _f(p.p4.E), "pt": _f(p.p4.pt),
             "eta": _f(p.p4.eta, 3), "flight_mm": _f(p.flight_length_mm, 6),
             "origin": origins[p.index]}
            for p in ev.particles if origins[p.index] in ("hard", "isr")
        ],
    }
    return payload


# --------------------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------------------


class CollideRequest(BaseModel):
    beam: BeamKind = "pp"
    channel: str = "z_mumu"
    sqrt_s: float = Field(1000.0, ge=pe.SQRT_S_MIN, le=pe.SQRT_S_MAX)
    b_field_tesla: float = Field(3.8, ge=B_FIELD_RANGE[0], le=B_FIELD_RANGE[1])
    asymmetry: float = Field(0.0, ge=-ASYMMETRY_MAX, le=ASYMMETRY_MAX)
    include_background: bool = True  # pp soft underlying event (ignored for e⁺e⁻)
    signal_fraction: float = Field(1.0, ge=0.0, le=1.0)  # P(signal) vs the channel's background
    seed: int | None = None


class LuminosityRequest(BaseModel):
    run_id: str | None = None
    beam: BeamKind = "pp"
    channel: str = "z_mumu"
    sqrt_s: float = Field(1000.0, ge=pe.SQRT_S_MIN, le=pe.SQRT_S_MAX)
    b_field_tesla: float = Field(3.8, ge=B_FIELD_RANGE[0], le=B_FIELD_RANGE[1])
    asymmetry: float = Field(0.0, ge=-ASYMMETRY_MAX, le=ASYMMETRY_MAX)
    include_background: bool = True
    signal_fraction: float = Field(0.3, ge=0.0, le=1.0)
    n_events: int = Field(50, ge=1, le=MAX_EVENTS_PER_STEP)
    n_bins: int = Field(60, ge=10, le=200)
    fit_model: FitModel = "voigt"
    include_event: bool = True
    seed: int | None = None


PLAYER_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class MysteryNewRequest(BaseModel):
    b_field_tesla: float = Field(3.8, ge=B_FIELD_RANGE[0], le=B_FIELD_RANGE[1])
    channel: str | None = None  # force a pool channel (debug / tests / screenshots)
    seed: int | None = None
    player: str | None = Field(None, pattern=PLAYER_PATTERN)


class MysteryHintRequest(BaseModel):
    challenge_id: str
    level: Literal[1, 2] = 1


class MysteryGuessRequest(BaseModel):
    challenge_id: str
    # a Step-A category ("tautau", see /api/config -> mystery.final_states) or particle counts
    particles: str | dict[str, int] | list[str]
    channel: str
    player: str | None = Field(None, pattern=PLAYER_PATTERN)


def _check_channel(key: str, *, signal_only: bool = False) -> pe.Channel:
    ch = pe.CHANNELS.get(key)
    if ch is None:
        raise HTTPException(422, f"unknown channel {key!r}; choose from {sorted(pe.CHANNELS)}")
    if signal_only and ch.kind != "signal":
        raise HTTPException(422, f"{key!r} is a background process, not a signal channel")
    return ch


# --------------------------------------------------------------------------------------
# Luminosity runs (server-side state, bounded LRU)
# --------------------------------------------------------------------------------------


class _RunSlot:
    def __init__(self, run: lab.LuminosityRun, fingerprint: tuple):
        self.run = run
        self.fingerprint = fingerprint
        self.lock = threading.Lock()
        self.created = time.time()


_runs: OrderedDict[str, _RunSlot] = OrderedDict()
_runs_lock = threading.Lock()


def _run_fingerprint(req: LuminosityRequest) -> tuple:
    return (req.beam, req.channel, round(req.sqrt_s, 6), round(req.b_field_tesla, 6),
            round(req.asymmetry, 6), req.include_background, round(req.signal_fraction, 6),
            req.n_bins)


def _get_or_create_run(req: LuminosityRequest) -> tuple[str, _RunSlot, bool]:
    fp = _run_fingerprint(req)
    with _runs_lock:
        if req.run_id and req.run_id in _runs and _runs[req.run_id].fingerprint == fp:
            _runs.move_to_end(req.run_id)
            return req.run_id, _runs[req.run_id], False
    ch = _check_channel(req.channel, signal_only=True)
    info = channel_info(ch)
    if not info["luminosity"]:
        raise HTTPException(422, f"{ch.label} has no resonance mass window for a luminosity "
                                 "run (try Z, J/ψ or Higgs channels)")
    beam = make_beam(req.beam, req.sqrt_s, req.asymmetry)
    if beam.sqrt_s < min_sqrt_s(ch, beam.kind):
        raise HTTPException(422, f"√s = {beam.sqrt_s:g} GeV is below the {ch.label} threshold "
                                 f"({min_sqrt_s(ch, beam.kind):.4g} GeV for {beam.kind})")
    run = lab.LuminosityRun(ch.key, beam=beam, signal_fraction=req.signal_fraction,
                            n_bins=req.n_bins, seed=req.seed,
                            detector_config=ds.DetectorConfig(b_field_t=req.b_field_tesla))
    run.generator.underlying_event = req.include_background
    run_id = uuid.uuid4().hex[:12]
    slot = _RunSlot(run, fp)
    with _runs_lock:
        _runs[run_id] = slot
        while len(_runs) > MAX_LUMI_RUNS:
            _runs.popitem(last=False)
    return run_id, slot, True


def fit_payload(run: lab.LuminosityRun, fit: lab.FitResult) -> dict[str, Any]:
    h = run.histogram
    xs = np.linspace(h.lo, h.hi, FIT_CURVE_POINTS)
    w = h.bin_width
    s_true, b_true = run.truth_counts_in(fit.window) if fit.b_window == fit.b_window else (0, 0)
    z_a = fit.significance_asimov
    return _clean({
        "model": fit.model, "success": fit.success, "message": fit.message,
        "mass": fit.mass, "mass_err": fit.mass_err, "width": fit.width,
        "width_err": fit.width_err, "sigma": fit.params.get("sigma"),
        "sigma_err": fit.errors.get("sigma"),
        "n_signal": fit.n_signal, "n_signal_err": fit.errors.get("n_sig"),
        "chi2": fit.chi2, "ndf": fit.ndf, "window": list(fit.window),
        "s_window": fit.s_window, "b_window": fit.b_window,
        "significance": fit.significance, "significance_asimov": z_a,
        "truth_s_window": s_true, "truth_b_window": b_true,
        "discovery": bool(z_a == z_a and z_a >= 5.0),
        "curve": {"x": xs, "total": fit.density(xs) * w, "background": fit.background_density(xs) * w},
    })


def luminosity_payload(run_id: str, slot: _RunSlot, req: LuminosityRequest, new_run: bool,
                       step_ms: float, rep: tuple | None) -> dict[str, Any]:
    run = slot.run
    h = run.histogram
    recs = run.records
    fit = None
    if h.entries >= MIN_ENTRIES_FOR_FIT:
        res = run.fit(req.fit_model)
        if res.success:
            fit = fit_payload(run, res)
    res_spec = pe.pdg(run.signal.resonance)
    return {
        "run_id": run_id, "new_run": new_run,
        "channel": run.signal.key, "label": run.signal.label,
        "background_channel": run.background,
        "background_label": pe.CHANNELS[run.background].label if run.background else None,
        "beam": beam_info(run.beam), "b_field_t": run.detector.config.b_field_t,
        "final_state": run.final_state, "signal_fraction": run.signal_fraction,
        "pdg_mass": _f(res_spec.mass), "pdg_width": _f(res_spec.width, 9),
        "n_events": run.n_events, "step_events": req.n_events, "step_ms": _f(step_ms, 1),
        "n_selected": h.entries,
        "n_signal_true": sum(r.is_signal for r in recs),
        "n_background_true": sum(not r.is_signal for r in recs),
        "n_selected_signal_true": sum(r.is_signal and r.in_window for r in recs),
        "efficiency": _f(run.efficiency),
        "histogram": {"lo": h.lo, "hi": h.hi, "n_bins": h.n_bins, "bin_width": _f(h.bin_width),
                      "edges": [_f(e) for e in h.edges], "counts": h.counts.tolist()},
        "fit": fit,
        "discovery": bool(fit and fit["discovery"]),
        "event": event_payload(*rep) if (rep and req.include_event) else None,
    }


# --------------------------------------------------------------------------------------
# Mystery mode state
# --------------------------------------------------------------------------------------


def mystery_meta() -> dict[str, Any]:
    return {
        "pool": [{"key": k, "label": pe.CHANNELS[k].label, "parent": lab.PARENT_OF[k],
                  "category": lab.SIGNATURE_CATEGORY[k]} for k in lab.MYSTERY_POOL],
        "final_states": [{"key": k, "label": v} for k, v in lab.FINAL_STATE_CATEGORIES.items()],
        "particle_choices": ["electron", "muon", "photon", "neutrino"],
        "hint_costs": {str(k): v for k, v in HINT_COSTS.items()},
        "beam": beam_info(pe.BeamConfig("pp", 1000.0)),
        "displaced_d0_mm": lab.DISPLACED_D0_MM,
    }


class _MysterySlot:
    def __init__(self, game: lab.MysteryChallenge, player: str):
        self.game = game
        self.player = player
        self.hint_level = 0
        self.result: dict[str, Any] | None = None  # cached response once graded


_mystery_games: dict[float, lab.MysteryChallenge] = {}
_mystery_events: OrderedDict[str, _MysterySlot] = OrderedDict()
_players: OrderedDict[str, dict[str, Any]] = OrderedDict()
_mystery_lock = threading.Lock()


def _mystery_challenge(b_field: float, seed: int | None) -> lab.MysteryChallenge:
    key = round(b_field, 4)
    if seed is not None:  # reproducible one-off game
        return lab.MysteryChallenge(seed=seed, detector_config=ds.DetectorConfig(b_field_t=b_field))
    if key not in _mystery_games:
        _mystery_games[key] = lab.MysteryChallenge(
            detector_config=ds.DetectorConfig(b_field_t=b_field))
    return _mystery_games[key]


def _player_stats(player: str) -> dict[str, Any]:
    if player not in _players:
        _players[player] = {"player": player, "rounds": 0, "total_score": 0, "streak": 0,
                            "best_streak": 0, "perfect": 0, "history": []}
        while len(_players) > MAX_PLAYERS:
            _players.popitem(last=False)
    _players.move_to_end(player)
    return _players[player]


def _public_stats(st: dict[str, Any]) -> dict[str, Any]:
    out = dict(st)
    out["history"] = list(st["history"])
    out["average"] = round(st["total_score"] / st["rounds"], 1) if st["rounds"] else None
    return out


def _slot(challenge_id: str) -> _MysterySlot:
    with _mystery_lock:
        slot = _mystery_events.get(challenge_id)
    if slot is None or challenge_id not in slot.game.events:
        raise HTTPException(404, f"unknown or expired challenge {challenge_id!r}")
    return slot


def _hints_for(me: lab.MysteryEvent, level: int) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if level >= 1:
        out["1"] = list(me.clues.layer_hints)
    if level >= 2:
        out["2"] = list(me.clues.hints)
    return out


def _penalty(level: int) -> int:
    return sum(cost for lv, cost in HINT_COSTS.items() if lv <= level)


# --------------------------------------------------------------------------------------
# FastAPI application
# --------------------------------------------------------------------------------------

app = FastAPI(title="Pocket CERN Collider", version="1.0.0",
              description="Relativistic event generator + detector simulation + analysis lab.")

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(pe.KinematicsError)
async def _kinematics_error(_request: Request, exc: pe.KinematicsError):
    return JSONResponse(status_code=422, content={"detail": f"Kinematically forbidden: {exc}"})


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    if not INDEX_HTML.is_file():
        raise HTTPException(404, "static/index.html is missing")
    return FileResponse(INDEX_HTML, media_type="text/html",
                        headers={"Cache-Control": "no-cache"})


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "channels": len(pe.CHANNELS), "luminosity_runs": len(_runs)}


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    return config_payload()


@app.post("/api/collide")
def collide(req: CollideRequest) -> dict[str, Any]:
    ch = _check_channel(req.channel)
    beam = make_beam(req.beam, req.sqrt_s, req.asymmetry)
    seed = req.seed if req.seed is not None else int(np.random.SeedSequence().entropy % 2**63)
    rng = np.random.default_rng(seed)
    gen = pe.CollisionGenerator(beam, seed=int(rng.integers(2**63)),
                                underlying_event=req.include_background)
    det_sim = ds.Detector(ds.DetectorConfig(b_field_t=req.b_field_tesla),
                          seed=int(rng.integers(2**63)))
    key = ch.key
    if ch.kind == "signal" and ch.background and rng.uniform() >= req.signal_fraction:
        key = ch.background
    t0 = time.perf_counter()
    if key != ch.key and ch.background_window:
        ev = gen.generate(key, mass_window=ch.background_window)
    else:
        ev = gen.generate(key)
    det = det_sim.simulate(ev)
    fs = lab.FINAL_STATE_FOR_CHANNEL.get(ch.key, "auto")
    cand = lab.reconstruct(det, fs) or (lab.reconstruct(det, "auto") if fs != "auto" else None)
    payload = event_payload(det, cand)
    payload.update({"seed": seed, "requested_channel": ch.key,
                    "sim_ms": _f((time.perf_counter() - t0) * 1e3, 2)})
    return payload


@app.post("/api/luminosity/step")
def luminosity_step(req: LuminosityRequest) -> dict[str, Any]:
    run_id, slot, new_run = _get_or_create_run(req)
    with slot.lock:
        run = slot.run
        before = run.last_selected
        t0 = time.perf_counter()
        run.run(req.n_events)
        step_ms = (time.perf_counter() - t0) * 1e3
        rep = run.last_selected if run.last_selected is not before else run.last_event
        return luminosity_payload(run_id, slot, req, new_run, step_ms, rep)


@app.delete("/api/luminosity/{run_id}")
def luminosity_delete(run_id: str) -> dict[str, Any]:
    with _runs_lock:
        existed = _runs.pop(run_id, None) is not None
    return {"run_id": run_id, "deleted": existed}


@app.post("/api/mystery/new")
def mystery_new(req: MysteryNewRequest) -> dict[str, Any]:
    if req.channel is not None and req.channel not in lab.MYSTERY_POOL:
        raise HTTPException(422, f"channel must be one of {list(lab.MYSTERY_POOL)}")
    player = req.player or "guest"
    with _mystery_lock:
        game = _mystery_challenge(req.b_field_tesla, req.seed)
        me = game.new_event(req.channel)
        _mystery_events[me.challenge_id] = _MysterySlot(game, player)
        while len(_mystery_events) > MAX_MYSTERY_EVENTS:
            old_id, old = _mystery_events.popitem(last=False)
            old.game.events.pop(old_id, None)
        stats = _public_stats(_player_stats(player))
    cand = lab.reconstruct(me.detector_event, "auto", game.selection)
    return {
        "challenge_id": me.challenge_id,
        "clues": _clean(me.clues.evidence()),  # raw numbers only; interpretation costs points
        "event": event_payload(me.detector_event, cand, hide_truth=True),
        **mystery_meta(),
        "beam": beam_info(game.beam),
        "stats": stats,
    }


@app.post("/api/mystery/hint")
def mystery_hint(req: MysteryHintRequest) -> dict[str, Any]:
    slot = _slot(req.challenge_id)
    me = slot.game.events[req.challenge_id]
    with _mystery_lock:
        if slot.result is None:  # hints after the verdict are free
            slot.hint_level = max(slot.hint_level, req.level)
        level = max(slot.hint_level, req.level)
    return {"challenge_id": req.challenge_id, "level": level, "hints": _hints_for(me, level),
            "penalty": _penalty(slot.hint_level)}


@app.post("/api/mystery/guess")
def mystery_guess(req: MysteryGuessRequest) -> dict[str, Any]:
    slot = _slot(req.challenge_id)
    game = slot.game
    if slot.result is not None:  # one claim per event: replay the verdict, no re-scoring
        return {**slot.result, "already_graded": True}
    if req.channel not in game.pool:
        raise HTTPException(422, f"channel must be one of {list(game.pool)}")
    try:
        grade = game.grade(req.challenge_id, req.particles, req.channel)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    me = game.events[req.challenge_id]
    penalty = _penalty(slot.hint_level)
    final = max(0, grade.score - penalty)
    correct = grade.particles_correct and grade.channel_correct
    with _mystery_lock:
        st = _player_stats(req.player or slot.player)
        st["rounds"] += 1
        st["total_score"] += final
        st["streak"] = st["streak"] + 1 if correct else 0
        st["best_streak"] = max(st["best_streak"], st["streak"])
        st["perfect"] += final == 100
        st["history"] = (st["history"] + [{"challenge_id": req.challenge_id,
                                           "truth": grade.truth_channel, "guess": req.channel,
                                           "score": final}])[-10:]
        stats = _public_stats(st)
    cand = lab.reconstruct(me.detector_event, "auto", game.selection)
    result = {
        "challenge_id": req.challenge_id,
        "grade": _clean(dict(grade.__dict__)),
        "hint_level": slot.hint_level, "hint_penalty": penalty, "final_score": final,
        "correct": correct,
        "truth_category_label": lab.FINAL_STATE_CATEGORIES.get(grade.truth_category or ""),
        "hints": _hints_for(me, 2),
        "reveal": event_payload(me.detector_event, cand),
        "stats": stats,
    }
    slot.result = result
    return {**result, "already_graded": False}


@app.get("/api/mystery/stats")
def mystery_stats(player: str = "guest") -> dict[str, Any]:
    with _mystery_lock:
        if player not in _players:
            raise HTTPException(404, f"no games recorded for player {player!r}")
        return _public_stats(_players[player])


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"Pocket CERN Collider control room → http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
