"""
detector_sim.py — Pocket CERN Collider · ATLAS/CMS-style detector simulation
============================================================================

Takes a truth ``physics_engine.Event`` and returns what a cylindrical collider detector
would *see*: helical tracks with silicon hits, calorimeter tower deposits, muon-chamber
hits, reconstructed (resolution-smeared) physics objects and the missing transverse
energy.

Units: momenta / energies in GeV; **detector coordinates in metres** (physics_engine
vertices are in mm and are converted on entry); B in tesla.

Geometry (radial shells, ATLAS/CMS-inspired)
--------------------------------------------
    BEAM_PIPE  r < 0.03 m
    TRACKER    0.03 – 1.10 m   silicon pixel + strip layers at discrete radii
    ECAL       1.10 – 1.70 m   Δφ-segmented towers
    HCAL       1.70 – 2.80 m   Δφ-segmented towers
    MUON       2.80 – 4.50 m   muon stations

Acceptance is the pseudorapidity cone |η| < η_max (default 2.5): anything more forward
escapes down the beam pipe unmeasured.  Inside the cone the shells are treated as
(barrel-projected) cylinders; trajectory z-coordinates are exact.

Larmor helix in the axial field B_z
-----------------------------------
    R[m] = p_T[GeV] / (0.299792458 · |q| · B[T])

Parametrised by transverse arc length s from the production vertex (x₀, y₀, z₀):

    φ(s) = φ₀ − h·s/R,         h = sign(q·B_z)
    x(s) = x_c − h R sin φ(s),  y(s) = y_c + h R cos φ(s),   z(s) = z₀ + s·p_z/p_T
    (x_c, y_c) = (x₀ + h R sin φ₀,  y₀ − h R cos φ₀)

so a positive charge in B_z > 0 bends clockwise when viewed from +z (F = q v⃗ × B⃗).
Crossings with a layer of radius r are found analytically (circle–circle intersection);
"loopers" whose circle never reaches r (2R < r for a prompt track) simply curl inside.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from functools import cached_property
from enum import Enum
from typing import Any

import numpy as np

import physics_engine as pe
from physics_engine import Event, FourVector, GenParticle

# --------------------------------------------------------------------------------------
# Geometry & constants
# --------------------------------------------------------------------------------------

LARMOR_CONSTANT = 0.299792458  # GeV / (T·m):  p_T = 0.3 · |q| · B · R
MM_TO_M = 1e-3


class Layer(str, Enum):
    BEAM_PIPE = "beam_pipe"
    TRACKER = "tracker"
    ECAL = "ecal"
    HCAL = "hcal"
    MUON = "muon"


R_BEAM_PIPE = 0.03
R_TRACKER = (0.03, 1.10)
R_ECAL = (1.10, 1.70)
R_HCAL = (1.70, 2.80)
R_MUON = (2.80, 4.50)

LAYER_RADII_M: dict[Layer, tuple[float, float]] = {
    Layer.BEAM_PIPE: (0.0, R_BEAM_PIPE),
    Layer.TRACKER: R_TRACKER,
    Layer.ECAL: R_ECAL,
    Layer.HCAL: R_HCAL,
    Layer.MUON: R_MUON,
}

PIXEL_LAYERS_M = (0.044, 0.073, 0.102)
STRIP_LAYERS_M = (0.255, 0.339, 0.418, 0.498, 0.608, 0.692, 0.780, 0.868, 0.965, 1.080)
SILICON_LAYERS_M = PIXEL_LAYERS_M + STRIP_LAYERS_M
MUON_STATIONS_M = (3.00, 3.45, 3.90, 4.35)

# physics-object kinds
ELECTRON = "electron"
PHOTON = "photon"
MUON = "muon"
CHARGED_HADRON = "charged_hadron"
NEUTRAL_HADRON = "neutral_hadron"
OBJECT_KINDS = (ELECTRON, PHOTON, MUON, CHARGED_HADRON, NEUTRAL_HADRON)

# truth-particle interaction classes (adds the invisible ones)
NEUTRINO = "neutrino"
REMNANT = "remnant"

_HADRON_LIKE_IDS = {pe.PI_PLUS, pe.K_PLUS, pe.PROTON, pe.NEUTRON, 130}


def particle_class(pdg_id: int, charge: float) -> str:
    """How a final-state truth particle interacts with matter."""
    a = abs(pdg_id)
    if pdg_id == pe.REMNANT:
        return REMNANT
    if a in pe.NEUTRINO_IDS:
        return NEUTRINO
    if a == pe.PHOTON:
        return PHOTON
    if a == pe.ELECTRON:
        return ELECTRON
    if a == pe.MUON:
        return MUON
    return CHARGED_HADRON if charge != 0 else NEUTRAL_HADRON


def resolution_tracker_pt(pt: float, a: float = 0.01, b: float = 0.0005) -> float:
    """σ(p_T)/p_T = a ⊕ b·p_T."""
    return math.hypot(a, b * pt)


def resolution_calo(e: float, stochastic: float, constant: float) -> float:
    """σ(E)/E = stochastic/√E ⊕ constant."""
    return math.hypot(stochastic / math.sqrt(max(e, 1e-9)), constant)


@dataclass(frozen=True)
class DetectorConfig:
    b_field_t: float = 3.8
    eta_max: float = 2.5
    ecal_phi_towers: int = 72  # Δφ = 5°
    hcal_phi_towers: int = 36  # Δφ = 10°
    tower_deta: float = 0.1
    # resolutions
    tracker_a: float = 0.01
    tracker_b: float = 0.0005
    ecal_stochastic: float = 0.10
    ecal_constant: float = 0.01
    hcal_stochastic: float = 0.50
    hcal_constant: float = 0.03
    # matter interaction
    hadron_ecal_fraction: tuple[float, float] = (0.05, 0.10)
    muon_ecal_mip_gev: float = 0.3
    muon_hcal_mip_gev: float = 1.5
    # reconstruction thresholds
    min_track_hits: int = 3
    min_track_pt: float = 0.1
    photon_min_e: float = 0.5
    neutral_hadron_min_e: float = 1.0
    # rendering
    polyline_step_m: float = 0.02
    looper_max_turns: float = 1.25


# --------------------------------------------------------------------------------------
# Helix
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Helix:
    """Charged-particle trajectory in a uniform axial field (straight line if q·B = 0)."""

    x0: float
    y0: float
    z0: float
    pt: float
    phi0: float
    cot_theta: float  # p_z / p_T
    charge: float
    b_field_t: float

    @classmethod
    def from_particle(cls, p: GenParticle, b_field_t: float) -> Helix:
        x, y, z = (c * MM_TO_M for c in p.production_vertex)
        pt = p.p4.pt
        return cls(x, y, z, pt, p.p4.phi, p.p4.pz / pt if pt > 0 else math.inf,
                   p.charge, b_field_t)

    @cached_property
    def h(self) -> int:
        """Rotation sense: +1 clockwise (viewed from +z), −1 counter-clockwise, 0 straight."""
        if self.charge == 0 or self.b_field_t == 0:
            return 0
        return 1 if self.charge * self.b_field_t > 0 else -1

    @cached_property
    def radius(self) -> float:
        """Larmor radius in metres."""
        if self.h == 0:
            return math.inf
        return self.pt / (LARMOR_CONSTANT * abs(self.charge) * abs(self.b_field_t))

    @cached_property
    def center(self) -> tuple[float, float]:
        h, r = self.h, self.radius
        return (self.x0 + h * r * math.sin(self.phi0), self.y0 - h * r * math.cos(self.phi0))

    @cached_property
    def rho0(self) -> float:
        return math.hypot(self.x0, self.y0)

    @cached_property
    def max_radius(self) -> float:
        """Largest transverse distance from the beam line the trajectory ever reaches."""
        if self.h == 0:
            return math.inf
        xc, yc = self.center
        return math.hypot(xc, yc) + self.radius

    def phi_at(self, s):
        if self.h == 0:
            return np.full_like(np.asarray(s, dtype=float), self.phi0)
        return self.phi0 - self.h * np.asarray(s, dtype=float) / self.radius

    def points(self, s) -> np.ndarray:
        """(N, 3) positions in metres at transverse arc lengths s."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        if self.h == 0:
            x = self.x0 + s * math.cos(self.phi0)
            y = self.y0 + s * math.sin(self.phi0)
        else:
            xc, yc = self.center
            hr = self.h * self.radius
            phi = self.phi_at(s)
            x = xc - hr * np.sin(phi)
            y = yc + hr * np.cos(phi)
        z = self.z0 + s * self.cot_theta
        return np.column_stack([x, y, z])

    def point_at(self, s: float) -> tuple[float, float, float]:
        """Scalar fast path of ``points`` for a single arc length."""
        if self.h == 0:
            return (self.x0 + s * math.cos(self.phi0), self.y0 + s * math.sin(self.phi0),
                    self.z0 + s * self.cot_theta)
        xc, yc = self.center
        hr = self.h * self.radius
        phi = self.phi0 - self.h * s / self.radius
        return (xc - hr * math.sin(phi), yc + hr * math.cos(phi), self.z0 + s * self.cot_theta)

    def arc_to_radius(self, r: float) -> float | None:
        """Smallest transverse arc length s > 0 at which the track reaches radius r.

        Only outward crossings (r > ρ₀) are considered; returns None for loopers that
        never get there (no math-domain errors: the circle–circle test is done first).
        """
        if r <= self.rho0:
            return None
        if self.h == 0:
            b = self.x0 * math.cos(self.phi0) + self.y0 * math.sin(self.phi0)
            c = self.rho0**2 - r * r  # < 0
            return -b + math.sqrt(b * b - c)
        R = self.radius
        xc, yc = self.center
        d = math.hypot(xc, yc)
        if d < 1e-15 or r > R + d or r < abs(R - d):
            return None
        a = (r * r - R * R + d * d) / (2.0 * d)
        hl = math.sqrt(max(r * r - a * a, 0.0))
        ux, uy = xc / d, yc / d
        best = None
        hr = self.h * R
        two_pi_r = 2.0 * math.pi * R
        for sgn in (1.0, -1.0):
            px, py = a * ux - sgn * hl * uy, a * uy + sgn * hl * ux
            phi = math.atan2(-(px - xc) / hr, (py - yc) / hr)
            s = (self.h * (self.phi0 - phi) * R) % two_pi_r
            if s > 1e-12 and (best is None or s < best):
                best = s
        return best


# --------------------------------------------------------------------------------------
# Detector records
# --------------------------------------------------------------------------------------


@dataclass
class CaloDeposit:
    layer: Layer
    energy: float  # measured (smeared) GeV
    true_energy: float
    phi: float  # impact azimuth
    eta: float
    iphi: int
    ieta: int
    position_m: tuple[float, float, float]  # impact point on the layer's inner face
    truth_index: int


@dataclass
class Track:
    truth_index: int
    helix: Helix
    charge_true: float
    charge_measured: float
    pt_true: float
    pt_measured: float
    hits: list[tuple[float, float, float]]  # silicon hits (m)
    muon_hits: list[tuple[float, float, float]]
    reconstructed: bool
    is_looper: bool

    @property
    def radius_m(self) -> float:
        return self.helix.radius


@dataclass
class ParticleTrace:
    """Everything that happened to one final-state truth particle in the detector."""

    truth_index: int
    pdg_id: int
    name: str
    particle_class: str
    fate: str  # invisible | escaped_forward | stopped_ecal | stopped_hcal | exited_muon | ...
    path: np.ndarray  # (N, 3) metres — charged: helix polyline; neutral: straight line
    visible_track: bool  # True if it leaves tracker hits
    layers_reached: list[Layer]
    track: Track | None = None
    deposits: list[CaloDeposit] = field(default_factory=list)

    @property
    def stopped_in(self) -> Layer | None:
        return {"stopped_ecal": Layer.ECAL, "stopped_hcal": Layer.HCAL}.get(self.fate)


@dataclass
class DetectedObject:
    kind: str
    p4: FourVector
    charge: int
    truth_index: int
    track: Track | None = None
    deposits: list[CaloDeposit] = field(default_factory=list)

    @property
    def pt(self) -> float:
        return self.p4.pt

    @property
    def eta(self) -> float:
        return self.p4.eta

    @property
    def phi(self) -> float:
        return self.p4.phi

    @property
    def energy(self) -> float:
        return self.p4.E


@dataclass
class DetectorEvent:
    truth: Event
    config: DetectorConfig
    traces: list[ParticleTrace]
    objects: list[DetectedObject]
    deposits: list[CaloDeposit]
    met_x: float = 0.0
    met_y: float = 0.0

    # ---- MET -------------------------------------------------------------------------
    @property
    def met(self) -> float:
        return math.hypot(self.met_x, self.met_y)

    @property
    def met_phi(self) -> float:
        return math.atan2(self.met_y, self.met_x)

    # ---- selections ------------------------------------------------------------------
    def objects_of(self, *kinds: str, min_pt: float = 0.0) -> list[DetectedObject]:
        out = [o for o in self.objects if o.kind in kinds and o.pt >= min_pt]
        return sorted(out, key=lambda o: o.pt, reverse=True)

    def leptons(self, min_pt: float = 0.0) -> list[DetectedObject]:
        return self.objects_of(ELECTRON, MUON, min_pt=min_pt)

    def trace_for(self, truth_index: int) -> ParticleTrace | None:
        return next((t for t in self.traces if t.truth_index == truth_index), None)

    def tower_energies(self, layer: Layer) -> np.ndarray:
        n = self.config.ecal_phi_towers if layer is Layer.ECAL else self.config.hcal_phi_towers
        e = np.zeros(n)
        for d in self.deposits:
            if d.layer is layer:
                e[d.iphi] += d.energy
        return e

    def layer_summary(self) -> dict[str, Any]:
        tracks = [t.track for t in self.traces if t.track and t.track.reconstructed]
        return {
            "n_tracks": len(tracks),
            "tracker_hits": sum(len(t.hits) for t in tracks),
            "ecal_energy": float(sum(d.energy for d in self.deposits if d.layer is Layer.ECAL)),
            "hcal_energy": float(sum(d.energy for d in self.deposits if d.layer is Layer.HCAL)),
            "muon_hits": sum(len(t.track.muon_hits) for t in self.traces if t.track),
            "met": self.met,
            "met_phi": self.met_phi,
            "n_escaped_forward": sum(t.fate == "escaped_forward" for t in self.traces),
            "n_invisible": sum(t.fate == "invisible" for t in self.traces),
        }

    def to_dict(self, decimals: int = 4) -> dict[str, Any]:
        """JSON-serialisable view (for the web UI). Contains NO Monte-Carlo truth labels
        beyond per-particle PDG ids on traces — callers hiding truth should drop 'traces'."""
        r = lambda v: round(float(v), decimals)  # noqa: E731
        return {
            "b_field_t": self.config.b_field_t,
            "met": r(self.met),
            "met_phi": r(self.met_phi),
            "objects": [
                {"kind": o.kind, "pt": r(o.pt), "eta": r(o.eta), "phi": r(o.phi),
                 "E": r(o.energy), "charge": o.charge}
                for o in self.objects
            ],
            "traces": [
                {"pdg_id": t.pdg_id, "name": t.name, "class": t.particle_class, "fate": t.fate,
                 "visible_track": t.visible_track,
                 "path": [[r(c) for c in pt] for pt in t.path],
                 "hits": [[r(c) for c in h] for h in (t.track.hits if t.track else [])],
                 "muon_hits": [[r(c) for c in h] for h in (t.track.muon_hits if t.track else [])]}
                for t in self.traces
            ],
            "towers": {
                "ecal": [r(e) for e in self.tower_energies(Layer.ECAL)],
                "hcal": [r(e) for e in self.tower_energies(Layer.HCAL)],
            },
        }


def exaggerate_displacement(point_m, primary_vertex_m, factor: float = 100.0) -> np.ndarray:
    """Display helper: magnify a (mm-scale) secondary-vertex offset from the primary vertex
    so that τ / K⁰_S decay points become visible on a metre-scale event display."""
    p = np.asarray(point_m, dtype=float)
    pv = np.asarray(primary_vertex_m, dtype=float)
    return pv + factor * (p - pv)


# --------------------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------------------


class Detector:
    def __init__(self, config: DetectorConfig | None = None, seed: int | None = None):
        self.config = config or DetectorConfig()
        self.rng = np.random.default_rng(seed)

    # ---- smearing --------------------------------------------------------------------
    def smear_calo(self, e: float, layer: Layer) -> float:
        c = self.config
        if layer is Layer.ECAL:
            sig = resolution_calo(e, c.ecal_stochastic, c.ecal_constant)
        else:
            sig = resolution_calo(e, c.hcal_stochastic, c.hcal_constant)
        return max(0.0, e * (1.0 + sig * self.rng.normal()))

    def smear_track(self, pt: float, charge: float) -> tuple[float, float]:
        """Smear the measured curvature κ = q/p_T (Gaussian in 1/p_T, like a real tracker).
        Very stiff tracks can come back with the wrong charge sign."""
        sig = resolution_tracker_pt(pt, self.config.tracker_a, self.config.tracker_b)
        kappa = (charge / pt) * (1.0 + sig * self.rng.normal())
        if kappa == 0.0:
            kappa = 1e-12
        return abs(charge / kappa) if charge else pt, math.copysign(abs(charge), kappa)

    # ---- geometry helpers -------------------------------------------------------------
    def _deposit(self, layer: Layer, true_e: float, pos: np.ndarray, eta: float,
                 truth_index: int, measured: float | None = None) -> CaloDeposit:
        c = self.config
        n = c.ecal_phi_towers if layer is Layer.ECAL else c.hcal_phi_towers
        phi = math.atan2(pos[1], pos[0])
        iphi = int(((phi % (2 * math.pi)) / (2 * math.pi)) * n) % n
        ieta = int(min(max((eta + c.eta_max) / c.tower_deta, 0), 2 * c.eta_max / c.tower_deta - 1))
        e = self.smear_calo(true_e, layer) if measured is None else measured
        return CaloDeposit(layer, e, true_e, phi, eta, iphi, ieta,
                           (float(pos[0]), float(pos[1]), float(pos[2])), truth_index)

    def _sample_path(self, helix: Helix, s_end: float) -> np.ndarray:
        if helix.h == 0:
            return helix.points([0.0, s_end])
        n = int(min(max(math.ceil(s_end / self.config.polyline_step_m), 8), 400))
        return helix.points(np.linspace(0.0, s_end, n + 1))

    def _crossings(self, helix: Helix, radii, s_end: float) -> list[tuple[float, float, float]]:
        out = []
        for r in radii:
            s = helix.arc_to_radius(r)
            if s is not None and s <= s_end:
                out.append(helix.point_at(s))
        return out

    # ---- main entry ------------------------------------------------------------------
    def simulate(self, event: Event) -> DetectorEvent:
        traces: list[ParticleTrace] = []
        objects: list[DetectedObject] = []
        for p in event.final_state():
            trace, obj = self._propagate(p)
            traces.append(trace)
            if obj is not None:
                objects.append(obj)
        deposits = [d for t in traces for d in t.deposits]
        det = DetectorEvent(event, self.config, traces, objects, deposits)
        det.met_x = -math.fsum(o.p4.px for o in objects)
        det.met_y = -math.fsum(o.p4.py for o in objects)
        return det

    def _propagate(self, p: GenParticle) -> tuple[ParticleTrace, DetectedObject | None]:
        c = self.config
        cls = particle_class(p.pdg_id, p.charge)
        vtx = np.array(p.production_vertex) * MM_TO_M
        no_path = vtx.reshape(1, 3)

        def trace(fate, path=no_path, visible=False, layers=(), track=None, deposits=()):
            return ParticleTrace(p.index, p.pdg_id, p.name, cls, fate, path, visible,
                                 list(layers), track, list(deposits))

        if cls == REMNANT:
            return trace("escaped_beam_remnant"), None
        if cls == NEUTRINO:
            # invisible: drawn (if at all) only as a straight line by the UI
            helix = Helix.from_particle(p, 0.0)
            s = helix.arc_to_radius(R_MUON[1]) or 0.0
            return trace("invisible", helix.points([0.0, s])), None
        eta = p.p4.eta
        if not abs(eta) < c.eta_max or p.p4.pt == 0.0:
            return trace("escaped_forward"), None

        if cls in (PHOTON, NEUTRAL_HADRON):
            return self._propagate_neutral(p, cls, eta, trace)
        return self._propagate_charged(p, cls, eta, trace)

    def _propagate_neutral(self, p: GenParticle, cls: str, eta: float, trace):
        c = self.config
        helix = Helix.from_particle(p, 0.0)  # straight line
        layer = Layer.ECAL if cls == PHOTON else Layer.HCAL
        r_face = LAYER_RADII_M[layer][0]
        s = helix.arc_to_radius(r_face)
        s = 0.0 if s is None else s
        impact = helix.points(s)[0]
        dep = self._deposit(layer, p.p4.E, impact, eta, p.index)
        layers = [Layer.TRACKER, Layer.ECAL] + ([Layer.HCAL] if layer is Layer.HCAL else [])
        tr = trace(f"stopped_{layer.value}", helix.points([0.0, s]), False, layers, None, [dep])
        threshold = c.photon_min_e if cls == PHOTON else c.neutral_hadron_min_e
        if dep.energy < threshold:
            return tr, None
        return tr, self._calo_object(cls, p, dep.energy, [dep])

    def _calo_object(self, kind: str, p: GenParticle, e: float, deps) -> DetectedObject:
        u = p.p4.p3 / p.p4.p  # direction from the production vertex
        return DetectedObject(kind, FourVector(e, *(e * u)), 0, p.index, None, list(deps))

    def _propagate_charged(self, p: GenParticle, cls: str, eta: float, trace):
        c = self.config
        rng = self.rng
        helix = Helix.from_particle(p, c.b_field_t)
        s_ecal = helix.arc_to_radius(R_ECAL[0])
        s_hcal = helix.arc_to_radius(R_HCAL[0])
        s_muon_in = helix.arc_to_radius(R_MUON[0])
        s_muon_out = helix.arc_to_radius(R_MUON[1])
        looper_end = c.looper_max_turns * 2 * math.pi * helix.radius

        # where does the particle stop?
        if cls == ELECTRON:
            s_end, fate = (s_ecal, "stopped_ecal") if s_ecal is not None else (looper_end, "looper")
        elif cls == MUON:
            if s_muon_out is not None:
                s_end, fate = s_muon_out, "exited_muon"
            elif s_muon_in is not None:  # enters the muon system, curls back before exiting
                s_end, fate = math.pi * helix.radius, "stopped_muon"
            elif s_ecal is not None:  # ranges out / curls up inside the calorimeters
                s_end, fate = math.pi * helix.radius, "curled_in_calo"
            else:
                s_end, fate = looper_end, "looper"
        else:  # charged hadron
            if s_hcal is not None:
                s_end, fate = s_hcal, "stopped_hcal"
            elif s_ecal is not None:
                s_end, fate = s_ecal, "stopped_hcal"  # showers at the calorimeter face
            else:
                s_end, fate = looper_end, "looper"

        path = self._sample_path(helix, s_end)
        hits = self._crossings(helix, SILICON_LAYERS_M, s_end)
        muon_hits = self._crossings(helix, MUON_STATIONS_M, s_end) if cls == MUON else []
        pt_m, q_m = self.smear_track(p.p4.pt, p.charge)
        reco = len(hits) >= c.min_track_hits and p.p4.pt >= c.min_track_pt
        track = Track(p.index, helix, p.charge, q_m, p.p4.pt, pt_m, hits, muon_hits,
                      reco, fate == "looper")

        layers = [Layer.TRACKER]
        deposits: list[CaloDeposit] = []
        E = p.p4.E
        if s_ecal is not None:
            layers.append(Layer.ECAL)
            at_ecal = np.array(helix.point_at(s_ecal))
            if cls == ELECTRON:
                deposits.append(self._deposit(Layer.ECAL, E, at_ecal, eta, p.index))
            elif cls == MUON:
                # MIP ionisation: Landau-like (gamma) fluctuations, not shower statistics
                mip = min(E, rng.gamma(4.0, c.muon_ecal_mip_gev / 4))
                deposits.append(self._deposit(Layer.ECAL, mip, at_ecal, eta, p.index, mip))
                if s_hcal is not None:
                    layers.append(Layer.HCAL)
                    mip = min(E, rng.gamma(4.0, c.muon_hcal_mip_gev / 4))
                    deposits.append(self._deposit(Layer.HCAL, mip, np.array(helix.point_at(s_hcal)),
                                                  eta, p.index, mip))
                if s_muon_in is not None:
                    layers.append(Layer.MUON)
            else:  # charged hadron: MIP-like fraction in ECAL, shower in HCAL
                f = rng.uniform(*c.hadron_ecal_fraction)
                deposits.append(self._deposit(Layer.ECAL, f * E, at_ecal, eta, p.index))
                at_hcal = np.array(helix.point_at(s_hcal)) if s_hcal is not None else at_ecal
                layers.append(Layer.HCAL)
                deposits.append(self._deposit(Layer.HCAL, (1 - f) * E, at_hcal, eta, p.index))

        tr = trace(fate, path, len(hits) > 0, layers, track, deposits)
        return tr, self._charged_object(p, cls, eta, track, deposits, s_ecal is not None)

    def _charged_object(self, p, cls, eta, track: Track, deposits, reached_ecal: bool):
        c = self.config
        if not track.reconstructed:
            # no usable track: whatever the calorimeters saw becomes a neutral object
            e_cal = sum(d.energy for d in deposits)
            if cls == ELECTRON and e_cal >= c.photon_min_e:
                return self._calo_object(PHOTON, p, e_cal, deposits)
            if cls == CHARGED_HADRON and e_cal >= c.neutral_hadron_min_e:
                return self._calo_object(NEUTRAL_HADRON, p, e_cal, deposits)
            return None
        q = int(round(track.charge_measured))
        if cls == ELECTRON and reached_ecal:
            e = max(deposits[0].energy, pe.pdg(pe.ELECTRON).mass * 1.000001)
            pm = math.sqrt(e * e - pe.pdg(pe.ELECTRON).mass ** 2)
            u = p.p4.p3 / p.p4.p
            return DetectedObject(ELECTRON, FourVector(e, *(pm * u)), q, p.index, track, deposits)
        if cls == MUON and reached_ecal:
            v = FourVector.from_pt_eta_phi_m(track.pt_measured, eta, p.p4.phi,
                                             pe.pdg(pe.MUON).mass)
            return DetectedObject(MUON, v, q, p.index, track, deposits)
        # charged hadrons, and e/μ that never reached the calorimeter (unidentifiable)
        v = FourVector.from_pt_eta_phi_m(track.pt_measured, eta, p.p4.phi,
                                         pe.pdg(pe.PI_PLUS).mass)
        return DetectedObject(CHARGED_HADRON, v, q, p.index, track, deposits)


# --------------------------------------------------------------------------------------
# Self-test / showcase
# --------------------------------------------------------------------------------------


def describe(det: DetectorEvent, max_rows: int = 40) -> str:
    lines = [
        f"{'#':>3} {'particle':<14}{'class':<15}{'pT':>8}{'eta':>7}  {'fate':<20}"
        f"{'R[m]':>8}{'hits':>5}{'mu':>4}  {'ECAL':>8}{'HCAL':>8}  object",
    ]
    obj_by_idx = {o.truth_index: o for o in det.objects}
    for t in det.traces[:max_rows]:
        p = det.truth.particles[t.truth_index]
        ecal = sum(d.energy for d in t.deposits if d.layer is Layer.ECAL)
        hcal = sum(d.energy for d in t.deposits if d.layer is Layer.HCAL)
        R = t.track.radius_m if t.track else math.inf
        o = obj_by_idx.get(t.truth_index)
        eta = p.p4.eta
        lines.append(
            f"{t.truth_index:>3} {t.name:<14}{t.particle_class:<15}{p.p4.pt:8.2f}"
            f"{eta if abs(eta) < 99 else float('nan'):7.2f}  {t.fate:<20}"
            f"{R if R < 1e4 else float('inf'):8.2f}{len(t.track.hits) if t.track else 0:>5}"
            f"{len(t.track.muon_hits) if t.track else 0:>4}  {ecal:8.2f}{hcal:8.2f}  "
            f"{(o.kind + f' pT={o.pt:.2f}') if o else '-'}"
        )
    s = det.layer_summary()
    lines.append(
        f"tracks={s['n_tracks']}  tracker hits={s['tracker_hits']}  "
        f"ECAL={s['ecal_energy']:.2f} GeV  HCAL={s['hcal_energy']:.2f} GeV  "
        f"muon hits={s['muon_hits']}  MET={det.met:.2f} GeV @ phi={det.met_phi:+.2f}"
    )
    return "\n".join(lines)


def run_self_test(seed: int = 7) -> bool:
    ok = True
    print("=" * 100)
    print(f"POCKET CERN COLLIDER - detector_sim self-test (B = {DetectorConfig().b_field_t} T)")
    print("=" * 100)
    det = Detector(seed=seed)
    showcase = [
        ("Z0 -> mu+ mu-", pe.BeamConfig("ee", 91.19), "z_mumu", {}),
        ("Z0 -> e+ e-", pe.BeamConfig("ee", 91.19), "z_ee", {}),
        ("H -> gamma gamma", pe.BeamConfig("ee", 126.0), "h_gammagamma", {}),
        ("W+W- -> l nu l nu", pe.BeamConfig("ee", 200.0), "ww_lnulnu", {}),
        ("Z0 -> tau+ tau-", pe.BeamConfig("ee", 91.19), "z_tautau", {}),
    ]
    for title, beam, ch, opts in showcase:
        ev = pe.CollisionGenerator(beam, seed=seed).generate(ch, **opts)
        d = det.simulate(ev)
        print(f"\n--- {title}  ({beam.label()}) ---")
        print(describe(d))
        if ch == "ww_lnulnu":
            mex, mey, met = ev.truth_missing_pt()
            print(f"truth neutrino pT = {met:.2f} GeV @ phi={math.atan2(mey, mex):+.2f}")

    # quick statistical rule checks
    print("\nStopping-rule audit over 300 pp @ 1 TeV events (all channels mixed):")
    gen = pe.CollisionGenerator(pe.BeamConfig("pp", 1000.0), seed=seed)
    counts: dict[tuple[str, str], int] = {}
    violations = 0
    keys = ["z_mumu", "z_ee", "h_gammagamma", "ww_lnulnu", "z_tautau", "h_zz_4l"]
    for i in range(300):
        d = det.simulate(gen.generate(keys[i % len(keys)]))
        for t in d.traces:
            counts[(t.particle_class, t.fate)] = counts.get((t.particle_class, t.fate), 0) + 1
            ecal = sum(x.energy for x in t.deposits if x.layer is Layer.ECAL)
            hcal = sum(x.energy for x in t.deposits if x.layer is Layer.HCAL)
            if t.particle_class == NEUTRINO and (t.deposits or t.visible_track):
                violations += 1
            if t.particle_class == PHOTON and (t.visible_track or hcal > 0):
                violations += 1
            if t.particle_class == ELECTRON and hcal > 0:
                violations += 1
            if t.particle_class == NEUTRAL_HADRON and (t.visible_track or ecal > 0):
                violations += 1
            if t.particle_class == MUON and t.track and t.track.muon_hits and \
                    t.fate not in ("exited_muon", "stopped_muon"):
                violations += 1
    for (cls, fate), n in sorted(counts.items()):
        print(f"  {cls:<16}{fate:<22}{n:>6}")
    print(f"  rule violations: {violations}")
    ok &= violations == 0
    print("\n" + ("ALL CHECKS PASSED" if ok else "SELF-TEST FAILED"))
    return ok


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(0 if run_self_test() else 1)
