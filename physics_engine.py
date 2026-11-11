"""
physics_engine.py — Pocket CERN Collider · event-generator core
================================================================

Exact relativistic Monte-Carlo kinematics in natural units:
energies / momenta / masses in GeV (c = ħ = 1), lengths in mm.
4-vectors are ordered (E, px, py, pz) with metric (+, −, −, −); the beam axis is ẑ.

How one collision is generated
------------------------------
1. **Beams.**  Two beam 4-vectors whose sum ``P_init`` has invariant mass √s.  An
   asymmetric collider is a CM frame boosted along ẑ by ``beta_z``.
2. **Hard process** (kinematics solved in the CM frame, then boosted to the lab):

   * e⁺e⁻ — the whole √s feeds the hard system.  If the sampled system mass M < √s the
     difference leaves as an initial-state-radiation (ISR) photon ("radiative return"),
     E_γ = (s − M²) / 2√s.
   * pp — partons carry momentum fractions x₁, x₂ solved *exactly* so the parton system
     has mass M at rapidity y; it gets a primordial k_T kick, soft underlying-event
     hadrons are added, and everything left over goes into two beam remnants.

3. **Decay chain.**  Resonance masses come from a truncated relativistic Breit–Wigner.
   Decays are exact 2-body (Källén momentum in the parent rest frame, then the 4×4
   Lorentz boost Λ(β⃗) to the lab) or flat 3-body phase space.  τ, ρ, π⁰ and K⁰_S decay
   automatically.  Every decay vertex is r⃗ = r⃗_prod + (p⃗/m)·cτ·ξ with ξ ~ Exp(1),
   i.e. βγ·c·t.

Exact 4-momentum conservation
-----------------------------
The last object of every split is obtained by *closure* (``P_parent − Σ others``) rather
than by an independent boost, and sums use ``math.fsum``.  Conservation therefore holds
to a few ulp of the beam energy (≲ 10⁻¹³ GeV at √s = 1 TeV) for arbitrarily deep cascades,
while the closure daughter remains on-shell to ~10⁻¹⁵ relative precision (verified in the
self-test and in ``tests/``).

Run ``python physics_engine.py`` for the self-test.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from functools import lru_cache
from typing import Callable, Iterable, Sequence

import numpy as np
from particle import Particle as _PDGParticle
from particle import pdgid as _pdgid

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

MEV_TO_GEV = 1e-3
HBARC_GEV_MM = 1.973269804e-13  # ħc in GeV·mm  (cτ = ħc / Γ)
ALPHA_EM = 1.0 / 137.035999084

SQRT_S_MIN = 2.0  # GeV
SQRT_S_MAX = 1000.0  # GeV

# PDG Monte-Carlo particle numbers
ELECTRON, NU_E, MUON, NU_MU, TAU, NU_TAU = 11, 12, 13, 14, 15, 16
PHOTON, Z0, W_PLUS, HIGGS = 22, 23, 24, 25
PI_0, PI_PLUS, RHO_PLUS, K_SHORT, K_PLUS = 111, 211, 213, 310, 321
J_PSI, NEUTRON, PROTON = 443, 2112, 2212
REMNANT = 0  # pseudo-id for a pp beam remnant (not a PDG particle)

NEUTRINO_IDS = frozenset({NU_E, NU_MU, NU_TAU})
SELF_CONJUGATE = frozenset({PHOTON, Z0, HIGGS, PI_0, K_SHORT, J_PSI})
LEPTON_IDS = {"e": ELECTRON, "mu": MUON, "tau": TAU}

#: The key particles of the brief (both charge states where relevant).
KEY_PARTICLES: tuple[int, ...] = (
    ELECTRON, -ELECTRON, MUON, -MUON, TAU, -TAU, NU_E, NU_MU, NU_TAU,
    PHOTON, PI_PLUS, -PI_PLUS, PI_0, K_PLUS, -K_PLUS, PROTON, NEUTRON,
    J_PSI, Z0, W_PLUS, -W_PLUS, HIGGS,
)

MAX_EVENT_ATTEMPTS = 200


class KinematicsError(ValueError):
    """The requested process is kinematically impossible for this configuration."""


class _RetryEvent(Exception):
    """Internal: a stochastic choice failed kinematic constraints, try again."""


def conjugate(pid: int) -> int:
    """PDG id of the antiparticle."""
    return pid if pid in SELF_CONJUGATE else -pid


# --------------------------------------------------------------------------------------
# PDG database
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParticleSpec:
    """Static PDG properties, converted to natural units."""

    pdg_id: int
    name: str
    mass: float  # GeV
    width: float  # GeV
    charge: float  # units of e
    ctau_mm: float  # mean proper decay length (mm); inf if stable
    spin: float | None
    kind: str  # lepton | neutrino | gauge boson | higgs | meson | baryon | other

    @property
    def is_stable(self) -> bool:
        return math.isinf(self.ctau_mm)

    @property
    def is_neutrino(self) -> bool:
        return abs(self.pdg_id) in NEUTRINO_IDS


def _classify(pid: int) -> str:
    a = abs(pid)
    if a in NEUTRINO_IDS or a == 18:
        return "neutrino"
    if _pdgid.is_lepton(pid):
        return "lepton"
    if a == HIGGS:
        return "higgs"
    if _pdgid.is_gauge_boson_or_higgs(pid):
        return "gauge boson"
    if _pdgid.is_meson(pid):
        return "meson"
    if _pdgid.is_baryon(pid):
        return "baryon"
    return "other"


@lru_cache(maxsize=None)
def pdg(pdg_id: int) -> ParticleSpec:
    """Look up a particle in the official PDG tables (via Scikit-HEP ``particle``)."""
    p = _PDGParticle.from_pdgid(pdg_id)
    mass = float(p.mass or 0.0) * MEV_TO_GEV
    width = float(p.width or 0.0) * MEV_TO_GEV
    ctau = p.ctau
    if ctau is None:
        ctau = HBARC_GEV_MM / width if width > 0 else math.inf
    spin = float(Fraction(p.J)) if p.J is not None else None
    return ParticleSpec(
        pdg_id=pdg_id,
        name=p.name,
        mass=mass,
        width=width,
        charge=float(p.charge or 0),
        ctau_mm=float(ctau),
        spin=spin,
        kind=_classify(pdg_id),
    )


def pdg_catalog(ids: Iterable[int] = KEY_PARTICLES) -> dict[int, ParticleSpec]:
    return {i: pdg(i) for i in ids}


# --------------------------------------------------------------------------------------
# 4-vectors and Lorentz boosts
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FourVector:
    """Contravariant 4-momentum (E, px, py, pz) in GeV."""

    E: float = 0.0
    px: float = 0.0
    py: float = 0.0
    pz: float = 0.0

    # ---- construction ----------------------------------------------------------------
    @classmethod
    def from_array(cls, a: Sequence[float]) -> FourVector:
        return cls(float(a[0]), float(a[1]), float(a[2]), float(a[3]))

    @classmethod
    def from_p3_mass(cls, px: float, py: float, pz: float, mass: float) -> FourVector:
        return cls(math.sqrt(mass * mass + px * px + py * py + pz * pz), px, py, pz)

    @classmethod
    def from_pt_eta_phi_m(cls, pt: float, eta: float, phi: float, mass: float) -> FourVector:
        pz = pt * math.sinh(eta)
        return cls.from_p3_mass(pt * math.cos(phi), pt * math.sin(phi), pz, mass)

    @staticmethod
    def sum(vectors: Iterable[FourVector]) -> FourVector:
        """Compensated (exactly-rounded) component-wise sum."""
        vs = list(vectors)
        return FourVector(
            math.fsum(v.E for v in vs),
            math.fsum(v.px for v in vs),
            math.fsum(v.py for v in vs),
            math.fsum(v.pz for v in vs),
        )

    def to_array(self) -> np.ndarray:
        return np.array([self.E, self.px, self.py, self.pz], dtype=np.float64)

    # ---- algebra ---------------------------------------------------------------------
    def __add__(self, o: FourVector) -> FourVector:
        return FourVector(self.E + o.E, self.px + o.px, self.py + o.py, self.pz + o.pz)

    def __sub__(self, o: FourVector) -> FourVector:
        return FourVector(self.E - o.E, self.px - o.px, self.py - o.py, self.pz - o.pz)

    def __neg__(self) -> FourVector:
        return FourVector(-self.E, -self.px, -self.py, -self.pz)

    def __mul__(self, k: float) -> FourVector:
        return FourVector(k * self.E, k * self.px, k * self.py, k * self.pz)

    __rmul__ = __mul__

    def dot(self, o: FourVector) -> float:
        """Minkowski product p·q = E E' − p⃗·p⃗'."""
        return self.E * o.E - self.px * o.px - self.py * o.py - self.pz * o.pz

    # ---- kinematic quantities --------------------------------------------------------
    @property
    def p3(self) -> np.ndarray:
        return np.array([self.px, self.py, self.pz], dtype=np.float64)

    @property
    def p(self) -> float:
        return math.sqrt(self.px * self.px + self.py * self.py + self.pz * self.pz)

    @property
    def pt(self) -> float:
        return math.hypot(self.px, self.py)

    @property
    def mass2(self) -> float:
        p = self.p
        return (self.E - p) * (self.E + p)  # cancellation-safe for light particles

    @property
    def mass(self) -> float:
        """Invariant mass √(E² − |p⃗|²); negative sign flags a space-like vector."""
        m2 = self.mass2
        return math.copysign(math.sqrt(abs(m2)), m2)

    @property
    def phi(self) -> float:
        return math.atan2(self.py, self.px)

    @property
    def theta(self) -> float:
        return math.atan2(self.pt, self.pz)

    @property
    def eta(self) -> float:
        """Pseudorapidity η = −ln tan(θ/2)."""
        pt = self.pt
        if pt == 0.0:
            return math.copysign(math.inf, self.pz)
        return math.asinh(self.pz / pt)

    @property
    def rapidity(self) -> float:
        num, den = self.E + self.pz, self.E - self.pz
        if den <= 0.0:
            return math.inf
        if num <= 0.0:
            return -math.inf
        return 0.5 * math.log(num / den)

    @property
    def beta_vec(self) -> np.ndarray:
        return self.p3 / self.E

    def boost(self, beta: Sequence[float], gamma: float | None = None) -> FourVector:
        """Return Λ(β⃗)·p: this vector seen from a frame moving with −β⃗ (active boost)."""
        return FourVector.from_array(lorentz_boost_matrix(beta, gamma) @ self.to_array())


def lorentz_boost_matrix(beta: Sequence[float], gamma: float | None = None) -> np.ndarray:
    """Exact 4×4 pure Lorentz boost Λ(β⃗) acting on (E, px, py, pz).

        Λ⁰₀ = γ,   Λ⁰ᵢ = Λⁱ₀ = γβᵢ,   Λⁱⱼ = δᵢⱼ + (γ − 1) βᵢβⱼ / β²

    ``(γ−1)/β²`` is evaluated as ``γ²/(1+γ)`` which is numerically stable from β → 0 to
    β → 1.  Pass ``gamma = E/m`` when known: it is more precise than 1/√(1−β²) for
    ultra-relativistic particles.
    """
    b = np.asarray(beta, dtype=np.float64)
    b2 = float(b @ b)
    if b2 == 0.0:
        return np.eye(4)
    if gamma is None:
        if b2 >= 1.0:
            raise ValueError(f"|β| = {math.sqrt(b2)} ≥ 1")
        gamma = 1.0 / math.sqrt(1.0 - b2)
    lam = np.empty((4, 4), dtype=np.float64)
    lam[0, 0] = gamma
    lam[0, 1:] = gamma * b
    lam[1:, 0] = gamma * b
    lam[1:, 1:] = np.eye(3) + (gamma * gamma / (1.0 + gamma)) * np.outer(b, b)
    return lam


MINKOWSKI_METRIC = np.diag([1.0, -1.0, -1.0, -1.0])


# --------------------------------------------------------------------------------------
# Decay kinematics
# --------------------------------------------------------------------------------------


def kallen_momentum(M: float, m1: float, m2: float) -> float:
    """Daughter momentum in the rest frame of a parent of mass M → m1 + m2."""
    if M < m1 + m2:
        raise KinematicsError(f"M = {M:.6g} GeV below threshold m1 + m2 = {m1 + m2:.6g} GeV")
    lam = (M * M - (m1 + m2) ** 2) * (M * M - (m1 - m2) ** 2)
    return math.sqrt(max(lam, 0.0)) / (2.0 * M)


def two_body_decay(
    parent: FourVector, M: float, m1: float, m2: float, cos_theta: float, phi: float
) -> tuple[FourVector, FourVector]:
    """Exact 2-body decay; (cos θ, φ) is daughter-1's direction in the parent rest frame.

    Daughter 1 is boosted with Λ(β⃗ = p⃗/E, γ = E/M); daughter 2 is P − p₁ (closure).
    """
    p = kallen_momentum(M, m1, m2)
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    d1_rest = FourVector(
        math.sqrt(m1 * m1 + p * p),
        p * sin_theta * math.cos(phi),
        p * sin_theta * math.sin(phi),
        p * cos_theta,
    )
    d1 = d1_rest.boost(parent.beta_vec, parent.E / M)
    return d1, parent - d1


def sample_cos_theta(rng: np.random.Generator, distribution: str = "isotropic") -> float:
    """Polar-angle sampler in the parent rest frame (axis = beam axis ẑ).

    * ``isotropic``   — spin-0 parents, phase space.
    * ``1+cos2``      — vector boson / γ* → f f̄ from unpolarised e⁺e⁻ or qq̄.
    * ``sin2``        — longitudinally polarised vector boson.
    """
    if distribution == "isotropic":
        return float(rng.uniform(-1.0, 1.0))
    if distribution == "1+cos2":
        while True:
            c = rng.uniform(-1.0, 1.0)
            if rng.uniform(0.0, 2.0) < 1.0 + c * c:
                return float(c)
    if distribution == "sin2":
        while True:
            c = rng.uniform(-1.0, 1.0)
            if rng.uniform(0.0, 1.0) < 1.0 - c * c:
                return float(c)
    raise ValueError(f"unknown angular distribution {distribution!r}")


def three_body_decay(
    rng: np.random.Generator,
    parent: FourVector,
    M: float,
    masses: tuple[float, float, float],
) -> tuple[FourVector, FourVector, FourVector]:
    """Flat (constant |𝓜|²) 3-body phase space via Dalitz-plot rejection sampling.

    s₁₂ is accepted with weight ∝ the allowed s₂₃ range, then the decay proceeds as
    P → (12) + 3 and (12) → 1 + 2, each isotropic — exactly uniform in the Dalitz plot.
    """
    m1, m2, m3 = masses
    if M < m1 + m2 + m3:
        raise KinematicsError("3-body decay below threshold")
    s12_lo, s12_hi = (m1 + m2) ** 2, (M - m3) ** 2
    s23_lo, s23_hi = (m2 + m3) ** 2, (M - m1) ** 2
    while True:
        s12 = rng.uniform(s12_lo, s12_hi)
        s23 = rng.uniform(s23_lo, s23_hi)
        m12 = math.sqrt(s12)
        e2 = (s12 - m1 * m1 + m2 * m2) / (2.0 * m12)
        e3 = (M * M - s12 - m3 * m3) / (2.0 * m12)
        p2 = math.sqrt(max(e2 * e2 - m2 * m2, 0.0))
        p3 = math.sqrt(max(e3 * e3 - m3 * m3, 0.0))
        lo = (e2 + e3) ** 2 - (p2 + p3) ** 2
        hi = (e2 + e3) ** 2 - (p2 - p3) ** 2
        if lo <= s23 <= hi:
            break
    d3, x12 = two_body_decay(
        parent, M, m3, m12, rng.uniform(-1, 1), rng.uniform(0, 2 * math.pi)
    )
    d1, d2 = two_body_decay(x12, m12, m1, m2, rng.uniform(-1, 1), rng.uniform(0, 2 * math.pi))
    return d1, d2, d3


# --------------------------------------------------------------------------------------
# Breit–Wigner line shape
# --------------------------------------------------------------------------------------


def relativistic_breit_wigner_pdf(m, m0: float, width: float):
    """Relativistic BW in mass, normalised to 1 on the physical range m ∈ [0, ∞):

        f(m) = N · 2m · m₀Γ / ((m² − m₀²)² + m₀²Γ²),   N = 1 / (π/2 + atan(m₀/Γ))

    (the s-space Cauchy shape would otherwise leave ≈ Γ/(π m₀) of its weight at s < 0).
    """
    m = np.asarray(m, dtype=np.float64)
    mg = m0 * width
    norm = 1.0 / (0.5 * math.pi + math.atan(m0 / width))
    return norm * 2.0 * m * mg / ((m * m - m0 * m0) ** 2 + mg * mg)


def sample_breit_wigner_mass(
    rng: np.random.Generator, m0: float, width: float, m_min: float, m_max: float
) -> float:
    """Sample M from the relativistic Breit–Wigner, truncated *exactly* to [m_min, m_max].

    In s = M² the line shape is 1/((s − m₀²)² + m₀²Γ²), whose CDF is an arctangent, so
    inverse-CDF sampling over the clamped interval is exact and rejection-free.
    """
    m_min = max(m_min, 0.0)
    if not m_max > m_min:
        raise KinematicsError(
            f"empty mass window [{m_min:.6g}, {m_max:.6g}] GeV for resonance m0 = {m0:.6g} GeV"
        )
    if width <= 0.0:
        if m_min <= m0 <= m_max:
            return m0
        raise KinematicsError(f"stable particle mass {m0} outside [{m_min}, {m_max}]")
    mg = m0 * width
    u_lo = math.atan((m_min * m_min - m0 * m0) / mg)
    u_hi = math.atan((m_max * m_max - m0 * m0) / mg)
    s = m0 * m0 + mg * math.tan(rng.uniform(u_lo, u_hi))
    return min(max(math.sqrt(max(s, 0.0)), m_min), m_max)


# --------------------------------------------------------------------------------------
# Beams
# --------------------------------------------------------------------------------------

_DEFAULT_VERTEX_SIGMA_MM = {
    "ee": (0.010, 0.0005, 0.30),  # flat e+e- bunches
    "pp": (0.015, 0.015, 45.0),  # LHC-like luminous region
}


@dataclass(frozen=True)
class BeamConfig:
    """Colliding-beam setup.

    ``kind``    — ``"ee"`` (e⁻ along +z, e⁺ along −z) or ``"pp"``.
    ``sqrt_s``  — centre-of-mass energy in GeV, 2 ≤ √s ≤ 1000.
    ``beta_z``  — velocity of the CM frame along ẑ (0 = symmetric collider).
    """

    kind: str = "pp"
    sqrt_s: float = 1000.0
    beta_z: float = 0.0
    vertex_sigma_mm: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("ee", "pp"):
            raise ValueError(f"beam kind must be 'ee' or 'pp', got {self.kind!r}")
        if not SQRT_S_MIN <= self.sqrt_s <= SQRT_S_MAX:
            raise ValueError(f"√s must be in [{SQRT_S_MIN}, {SQRT_S_MAX}] GeV, got {self.sqrt_s}")
        if not abs(self.beta_z) < 1.0:
            raise ValueError("|beta_z| must be < 1")
        if self.sqrt_s <= 2.0 * self.beam_mass:
            raise ValueError("√s below 2 × beam-particle mass")
        if self.vertex_sigma_mm is None:
            object.__setattr__(self, "vertex_sigma_mm", _DEFAULT_VERTEX_SIGMA_MM[self.kind])

    @classmethod
    def from_beam_energies(cls, kind: str, e_beam1: float, e_beam2: float, **kw) -> BeamConfig:
        """Asymmetric collider from beam energies (beam 1 along +z, beam 2 along −z)."""
        m = pdg(ELECTRON if kind == "ee" else PROTON).mass
        b1 = FourVector(e_beam1, 0.0, 0.0, math.sqrt(e_beam1**2 - m * m))
        b2 = FourVector(e_beam2, 0.0, 0.0, -math.sqrt(e_beam2**2 - m * m))
        tot = b1 + b2
        return cls(kind=kind, sqrt_s=tot.mass, beta_z=tot.pz / tot.E, **kw)

    @property
    def s(self) -> float:
        return self.sqrt_s**2

    @property
    def beam_ids(self) -> tuple[int, int]:
        return (ELECTRON, -ELECTRON) if self.kind == "ee" else (PROTON, PROTON)

    @property
    def beam_mass(self) -> float:
        return pdg(self.beam_ids[0]).mass

    @property
    def initial_charge(self) -> float:
        return sum(pdg(i).charge for i in self.beam_ids)

    @property
    def gamma_z(self) -> float:
        return 1.0 / math.sqrt(1.0 - self.beta_z**2)

    def to_lab(self, v: FourVector) -> FourVector:
        if self.beta_z == 0.0:
            return v
        return v.boost((0.0, 0.0, self.beta_z), self.gamma_z)

    def cm_beams(self) -> tuple[FourVector, FourVector]:
        e = 0.5 * self.sqrt_s
        pb = math.sqrt(e * e - self.beam_mass**2)
        return FourVector(e, 0.0, 0.0, pb), FourVector(e, 0.0, 0.0, -pb)

    def lab_beams(self) -> tuple[FourVector, FourVector]:
        b1, b2 = self.cm_beams()
        return self.to_lab(b1), self.to_lab(b2)

    @property
    def initial_p4(self) -> FourVector:
        """Total initial 4-momentum P_init in the lab frame."""
        return FourVector.sum(self.lab_beams())

    def label(self) -> str:
        tag = "e+e-" if self.kind == "ee" else "pp"
        asym = f", beta_z={self.beta_z:+.3f}" if self.beta_z else ""
        return f"{tag} @ sqrt(s)={self.sqrt_s:.4g} GeV{asym}"


# --------------------------------------------------------------------------------------
# Event record
# --------------------------------------------------------------------------------------


class Status(str, Enum):
    FINAL = "final"  # stable, leaves the interaction region
    DECAYED = "decayed"  # intermediate / resonance
    REMNANT = "remnant"  # pp beam remnant (escapes down the beam pipe)


@dataclass
class GenParticle:
    index: int
    pdg_id: int
    name: str
    p4: FourVector
    mass: float  # generated (possibly off-shell) mass, GeV
    charge: float
    status: Status
    mother: int | None
    daughters: list[int] = field(default_factory=list)
    production_vertex: tuple[float, float, float] = (0.0, 0.0, 0.0)  # mm
    decay_vertex: tuple[float, float, float] | None = None  # mm
    ctau_mm: float = math.inf
    proper_decay_length_mm: float | None = None  # sampled cτ·ξ

    @property
    def is_final(self) -> bool:
        return self.status in (Status.FINAL, Status.REMNANT)

    @property
    def is_neutrino(self) -> bool:
        return abs(self.pdg_id) in NEUTRINO_IDS

    @property
    def is_invisible(self) -> bool:
        """Escapes detection: neutrinos and beam remnants."""
        return self.is_neutrino or self.status is Status.REMNANT

    @property
    def flight_length_mm(self) -> float:
        if self.decay_vertex is None:
            return math.inf if self.status is not Status.DECAYED else 0.0
        return math.dist(self.production_vertex, self.decay_vertex)


@dataclass
class Event:
    event_id: int
    channel: str
    beam: BeamConfig
    initial_p4: FourVector
    initial_charge: float
    particles: list[GenParticle]
    primary_vertex: tuple[float, float, float]
    true_mass: float  # mass of the hard system (resonance / continuum) in GeV
    is_signal: bool

    # ---- selections ------------------------------------------------------------------
    def final_state(self) -> list[GenParticle]:
        return [p for p in self.particles if p.is_final]

    def visible_final_state(self) -> list[GenParticle]:
        return [p for p in self.particles if p.is_final and not p.is_invisible]

    def neutrinos(self) -> list[GenParticle]:
        return [p for p in self.particles if p.is_final and p.is_neutrino]

    def find(self, *pdg_ids: int) -> list[GenParticle]:
        return [p for p in self.particles if p.pdg_id in pdg_ids]

    # ---- conservation ----------------------------------------------------------------
    def total_final_p4(self) -> FourVector:
        return FourVector.sum(p.p4 for p in self.final_state())

    def conservation_residual(self) -> np.ndarray:
        """P_init − Σ P_final  (all final particles, invisible ones included)."""
        fin = self.final_state()
        cols = [
            [self.initial_p4.E] + [-p.p4.E for p in fin],
            [self.initial_p4.px] + [-p.p4.px for p in fin],
            [self.initial_p4.py] + [-p.p4.py for p in fin],
            [self.initial_p4.pz] + [-p.p4.pz for p in fin],
        ]
        return np.array([math.fsum(c) for c in cols])

    def max_conservation_violation(self) -> float:
        return float(np.max(np.abs(self.conservation_residual())))

    def charge_residual(self) -> float:
        return self.initial_charge - math.fsum(p.charge for p in self.final_state())

    def truth_missing_pt(self) -> tuple[float, float, float]:
        """True (MEx, MEy, MET) carried by neutrinos."""
        mex = math.fsum(p.p4.px for p in self.neutrinos())
        mey = math.fsum(p.p4.py for p in self.neutrinos())
        return mex, mey, math.hypot(mex, mey)

    # ---- display ---------------------------------------------------------------------
    def table(self) -> str:
        head = (
            f"{'#':>3} {'particle':<16}{'status':<8}{'mom':>4} "
            f"{'E':>11}{'px':>11}{'py':>11}{'pz':>11}{'m':>10}"
            f"{'pT':>9}{'eta':>8}{'phi':>7}{'r_prod':>10}"
        )
        lines = [
            f"Event {self.event_id} | channel={self.channel} | {self.beam.label()} | "
            f"M_true={self.true_mass:.5f} GeV",
            head,
            "-" * len(head),
        ]
        for p in self.particles:
            v = p.p4
            eta = v.eta
            eta_s = f"{eta:8.3f}" if abs(eta) < 1e3 else f"{'±inf':>8}"
            r = math.hypot(*p.production_vertex)  # 3D distance from origin
            lines.append(
                f"{p.index:>3} {p.name:<16}{p.status.value[:7]:<8}"
                f"{'' if p.mother is None else p.mother:>4} "
                f"{v.E:11.5f}{v.px:11.5f}{v.py:11.5f}{v.pz:11.5f}{p.mass:10.5f}"
                f"{v.pt:9.4f}{eta_s}{v.phi:7.3f}{r:10.4f}"
            )
        res = self.conservation_residual()
        tot = self.total_final_p4()
        mex, mey, met = self.truth_missing_pt()
        lines += [
            "-" * len(head),
            f"P_init        = ({self.initial_p4.E:.10f}, {self.initial_p4.px:.10f}, "
            f"{self.initial_p4.py:.10f}, {self.initial_p4.pz:.10f}) GeV",
            f"Sum P_final   = ({tot.E:.10f}, {tot.px:.10f}, {tot.py:.10f}, {tot.pz:.10f}) GeV",
            f"Residual      = ({res[0]:.2e}, {res[1]:.2e}, {res[2]:.2e}, {res[3]:.2e}) GeV",
            f"Charge: initial {self.initial_charge:+.0f}, residual {self.charge_residual():+.0f}"
            f" | truth MET (neutrinos) = {met:.4f} GeV",
        ]
        return "\n".join(lines)


class _EventBuilder:
    """Mutable truth-record under construction (everything already in the lab frame)."""

    def __init__(self, rng: np.random.Generator, primary_vertex: tuple[float, float, float]):
        self.rng = rng
        self.primary_vertex = primary_vertex
        self.particles: list[GenParticle] = []

    def add(
        self,
        pid: int,
        p4: FourVector,
        *,
        mass: float | None = None,
        mother: int | None = None,
        name: str | None = None,
        charge: float | None = None,
        ctau_mm: float | None = None,
        status: Status = Status.FINAL,
    ) -> int:
        spec = pdg(pid) if pid != REMNANT else None
        if mother is None:
            vtx = self.primary_vertex
        else:
            vtx = self.particles[mother].decay_vertex
            self.particles[mother].daughters.append(len(self.particles))
        gp = GenParticle(
            index=len(self.particles),
            pdg_id=pid,
            name=name or (spec.name if spec else "remnant"),
            p4=p4,
            mass=(spec.mass if spec else 0.0) if mass is None else mass,
            charge=(spec.charge if spec else 0.0) if charge is None else charge,
            status=status,
            mother=mother,
            production_vertex=vtx,
            ctau_mm=(spec.ctau_mm if spec else math.inf) if ctau_mm is None else ctau_mm,
        )
        self.particles.append(gp)
        return gp.index

    def mark_decayed(self, idx: int) -> GenParticle:
        """Sample the proper decay length and place the decay vertex r = (p⃗/m)·cτ·ξ."""
        p = self.particles[idx]
        if p.status is Status.DECAYED:
            return p
        ct = p.ctau_mm if math.isfinite(p.ctau_mm) else 0.0
        length = float(self.rng.exponential(ct)) if ct > 0 else 0.0
        p.status = Status.DECAYED
        p.proper_decay_length_mm = length
        bg = p.p4.p3 / p.mass if p.mass > 0 else np.zeros(3)  # βγ vector
        x0, y0, z0 = p.production_vertex
        p.decay_vertex = (x0 + bg[0] * length, y0 + bg[1] * length, z0 + bg[2] * length)
        return p

    def split(
        self, idx: int, daughters: Sequence[tuple[int, FourVector, float | None]], **kw
    ) -> list[int]:
        """Attach already-computed daughter momenta (pid, p4, mass) to particle idx."""
        self.mark_decayed(idx)
        return [self.add(pid, v, mass=m, mother=idx, **kw) for pid, v, m in daughters]

    def decay_2body(
        self,
        idx: int,
        pid1: int,
        pid2: int,
        *,
        m1: float | None = None,
        m2: float | None = None,
        angular: str = "isotropic",
    ) -> tuple[int, int]:
        par = self.particles[idx]
        m1 = pdg(pid1).mass if m1 is None else m1
        m2 = pdg(pid2).mass if m2 is None else m2
        d1, d2 = two_body_decay(
            par.p4,
            par.mass,
            m1,
            m2,
            sample_cos_theta(self.rng, angular),
            self.rng.uniform(0.0, 2.0 * math.pi),
        )
        i1, i2 = self.split(idx, [(pid1, d1, m1), (pid2, d2, m2)])
        return i1, i2

    def decay_3body(self, idx: int, pids: tuple[int, int, int]) -> list[int]:
        par = self.particles[idx]
        ms = tuple(pdg(i).mass for i in pids)
        vs = three_body_decay(self.rng, par.p4, par.mass, ms)  # type: ignore[arg-type]
        return self.split(idx, [(pid, v, m) for pid, v, m in zip(pids, vs, ms)])


# --------------------------------------------------------------------------------------
# Automatic decays of short-lived final-state particles
# --------------------------------------------------------------------------------------

# τ⁻ decay modes (PDG branching fractions; renormalised over the modes we model)
_TAU_MODES: tuple[tuple[float, tuple[int, ...]], ...] = (
    (0.1782, (ELECTRON, -NU_E, NU_TAU)),
    (0.1739, (MUON, -NU_MU, NU_TAU)),
    (0.1082, (-PI_PLUS, NU_TAU)),
    (0.2549, (-RHO_PLUS, NU_TAU)),
)
_K_SHORT_MODES = ((0.692, (PI_PLUS, -PI_PLUS)), (0.307, (PI_0, PI_0)))
AUTO_DECAY_IDS = frozenset({TAU, -TAU, RHO_PLUS, -RHO_PLUS, PI_0, K_SHORT})


def _choose_mode(rng: np.random.Generator, modes) -> tuple[int, ...]:
    w = np.array([m[0] for m in modes])
    return modes[int(rng.choice(len(modes), p=w / w.sum()))][1]


def _auto_decay(b: _EventBuilder, idx: int) -> None:
    p = b.particles[idx]
    pid, rng = p.pdg_id, b.rng
    if abs(pid) == TAU:
        mode = _choose_mode(rng, _TAU_MODES)
        if pid < 0:
            mode = tuple(conjugate(i) for i in mode)
        if len(mode) == 3:
            b.decay_3body(idx, mode)  # type: ignore[arg-type]
        elif abs(mode[0]) == RHO_PLUS:
            rho = pdg(RHO_PLUS)
            m_rho = sample_breit_wigner_mass(
                rng, rho.mass, rho.width, pdg(PI_PLUS).mass + pdg(PI_0).mass, p.mass
            )
            b.decay_2body(idx, mode[0], mode[1], m1=m_rho)
        else:
            b.decay_2body(idx, mode[0], mode[1])
    elif abs(pid) == RHO_PLUS:
        b.decay_2body(idx, PI_PLUS if pid > 0 else -PI_PLUS, PI_0)
    elif pid == PI_0:
        b.decay_2body(idx, PHOTON, PHOTON)
    elif pid == K_SHORT:
        b.decay_2body(idx, *_choose_mode(rng, _K_SHORT_MODES))


def _decay_all(b: _EventBuilder) -> None:
    i = 0
    while i < len(b.particles):  # daughters are appended, so this walks the full cascade
        p = b.particles[i]
        if p.status is Status.FINAL and p.pdg_id in AUTO_DECAY_IDS:
            _auto_decay(b, i)
        i += 1


# --------------------------------------------------------------------------------------
# Channels
# --------------------------------------------------------------------------------------

ChannelFn = Callable[["CollisionGenerator", _EventBuilder], float]


@dataclass(frozen=True)
class Channel:
    key: str
    label: str
    kind: str  # "signal" | "background"
    resonance: int | None
    description: str
    recommended_sqrt_s: dict[str, float]
    fn: ChannelFn
    background: str | None = None
    background_window: tuple[float, float] | None = None  # GeV, for luminosity runs


CHANNELS: dict[str, Channel] = {}


def _channel(key, label, kind, resonance, description, recommended, background=None, window=None):
    def deco(fn: ChannelFn) -> ChannelFn:
        CHANNELS[key] = Channel(
            key, label, kind, resonance, description, recommended, fn, background, window
        )
        return fn

    return deco


# --------------------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------------------


class CollisionGenerator:
    """Generates fully-decayed truth events for a given beam configuration."""

    PP_KT_SIGMA = 2.0  # GeV, primordial k_T (Rayleigh σ)
    UE_PT_MEAN = 0.45  # GeV, soft underlying-event hadron <pT>
    UE_ETA_MAX = 4.0
    REMNANT_MASS_MARGIN = 0.05  # GeV above 2 m_p

    def __init__(self, beam: BeamConfig, seed: int | None = None, underlying_event: bool = True):
        self.beam = beam
        self.rng = np.random.default_rng(seed)
        self.underlying_event = underlying_event
        self._next_id = 0

    # ---- public ----------------------------------------------------------------------
    def generate(self, channel: str, **options) -> Event:
        if channel not in CHANNELS:
            raise KeyError(f"unknown channel {channel!r}; choose from {sorted(CHANNELS)}")
        ch = CHANNELS[channel]
        for attempt in range(MAX_EVENT_ATTEMPTS):
            b = _EventBuilder(self.rng, self._sample_primary_vertex())
            self._attempt = attempt
            try:
                true_mass = ch.fn(self, b, **options)
                _decay_all(b)
            except _RetryEvent:
                continue
            return self._finalize(b, ch, true_mass)
        raise KinematicsError(
            f"could not generate {channel} for {self.beam.label()} "
            f"after {MAX_EVENT_ATTEMPTS} attempts"
        )

    def generate_many(self, channel: str, n: int, **options) -> list[Event]:
        return [self.generate(channel, **options) for _ in range(n)]

    # ---- helpers used by channels ----------------------------------------------------
    @property
    def max_system_mass(self) -> float:
        """Largest hard-system mass reachable in this beam configuration."""
        if self.beam.kind == "ee":
            return self.beam.sqrt_s
        return self.beam.sqrt_s - 2.0 * pdg(PROTON).mass - 0.1

    def bw_mass(self, pid: int, m_min: float, m_max: float, require_peak: bool = False) -> float:
        spec = pdg(pid)
        if require_peak and m_max < spec.mass - 5.0 * spec.width:
            raise KinematicsError(
                f"{spec.name} (m0 = {spec.mass:.4f} GeV) is not reachable: maximum hard-system "
                f"mass is {m_max:.4f} GeV for {self.beam.label()}"
            )
        return sample_breit_wigner_mass(self.rng, spec.mass, spec.width, m_min, m_max)

    def continuum_mass(self, lo: float, hi: float, power: float = 3.0) -> float:
        """Non-resonant hard-system mass in [lo, hi].

        e⁺e⁻: ISR "radiator" spectrum in v = 1 − M²/s, density ∝ β v^(β−1) with
        β = (2α/π)(ln(s/m_e²) − 1) — most events sit at √s, with a radiative-return tail.
        pp:   steeply-falling parton-luminosity power law dN/dM ∝ M^(−power).
        """
        hi = min(hi, self.max_system_mass)
        if not hi > lo:
            raise KinematicsError(f"empty continuum window [{lo:.4g}, {hi:.4g}] GeV")
        r = self.rng.uniform()
        if self.beam.kind == "ee":
            s = self.beam.s
            beta = (2.0 * ALPHA_EM / math.pi) * (math.log(s / pdg(ELECTRON).mass ** 2) - 1.0)
            v_lo, v_hi = max(0.0, 1.0 - hi * hi / s), 1.0 - lo * lo / s
            v = (v_lo**beta + r * (v_hi**beta - v_lo**beta)) ** (1.0 / beta)
            return min(max(self.beam.sqrt_s * math.sqrt(1.0 - v), lo), hi)
        a = 1.0 - power
        return (lo**a + r * (hi**a - lo**a)) ** (1.0 / a)

    def default_window(self, threshold: float, window: tuple[float, float] | None):
        if window is not None:
            lo, hi = window
        else:
            lo, hi = 0.05 * self.beam.sqrt_s, self.max_system_mass
        return max(lo, threshold + 1e-6), hi

    def produce(self, b: _EventBuilder, M: float) -> tuple[FourVector, float]:
        """Create the beam-side recoil for a hard system of mass M; return (P_hard_lab, M)."""
        if M > self.max_system_mass:
            raise KinematicsError(f"M = {M:.4f} GeV exceeds reach of {self.beam.label()}")
        return self._produce_ee(b, M) if self.beam.kind == "ee" else self._produce_pp(b, M)

    # ---- production internals --------------------------------------------------------
    def _sample_primary_vertex(self) -> tuple[float, float, float]:
        sx, sy, sz = self.beam.vertex_sigma_mm  # type: ignore[misc]
        return (
            float(self.rng.normal(0, sx)),
            float(self.rng.normal(0, sy)),
            float(self.rng.normal(0, sz)),
        )

    def _produce_ee(self, b: _EventBuilder, M: float) -> tuple[FourVector, float]:
        beam = self.beam
        p_init = beam.initial_p4
        e_gamma = (beam.s - M * M) / (2.0 * beam.sqrt_s)  # ISR photon energy in CM
        if e_gamma < 1e-6:
            return p_init, beam.sqrt_s
        # ISR photons are strongly collinear with the beams: 1 − |cos θ| log-uniform
        t = 10.0 ** self.rng.uniform(-7.0, 0.0)
        c = (1.0 - t) * (1.0 if self.rng.uniform() < 0.5 else -1.0)
        st = math.sqrt(max(0.0, 1.0 - c * c))
        phi = self.rng.uniform(0.0, 2.0 * math.pi)
        g_cm = FourVector(e_gamma, e_gamma * st * math.cos(phi), e_gamma * st * math.sin(phi),
                          e_gamma * c)
        g_lab = beam.to_lab(g_cm)
        b.add(PHOTON, g_lab, name="gamma (ISR)")
        return p_init - g_lab, M

    def _underlying_event(self) -> list[tuple[int, FourVector]]:
        mean = max(0.0, 2.2 * math.log(self.beam.sqrt_s / 5.0))
        species = ((PI_PLUS, 0.35), (-PI_PLUS, 0.35), (PI_0, 0.18), (K_PLUS, 0.04),
                   (-K_PLUS, 0.04), (K_SHORT, 0.04))
        ids = [s[0] for s in species]
        w = np.array([s[1] for s in species])
        out = []
        for _ in range(int(self.rng.poisson(mean))):
            pid = ids[int(self.rng.choice(len(ids), p=w / w.sum()))]
            pt = float(self.rng.exponential(self.UE_PT_MEAN)) + 0.05
            eta = float(self.rng.uniform(-self.UE_ETA_MAX, self.UE_ETA_MAX))
            phi = float(self.rng.uniform(0, 2 * math.pi))
            out.append((pid, FourVector.from_pt_eta_phi_m(pt, eta, phi, pdg(pid).mass)))
        return out

    def _produce_pp(self, b: _EventBuilder, M: float) -> tuple[FourVector, float]:
        beam = self.beam
        m_p = pdg(PROTON).mass
        e = 0.5 * beam.sqrt_s
        pb = math.sqrt(e * e - m_p * m_p)
        y_max = math.log(beam.sqrt_s / M)
        for _ in range(50):
            # parton momentum fractions: (x1 p1 + x2 p2)² = M² solved exactly for given y
            y = float(self.rng.normal(0.0, max(0.5 * y_max, 1e-9)))
            if abs(y) > y_max:
                continue
            den = e * e * math.cosh(y) ** 2 - pb * pb * math.sinh(y) ** 2
            a = M / (2.0 * math.sqrt(den))
            x1, x2 = a * math.exp(y), a * math.exp(-y)
            if x1 >= 1.0 or x2 >= 1.0:
                continue
            kt = float(self.rng.rayleigh(min(self.PP_KT_SIGMA, 0.1 * M + 0.2)))
            phi = float(self.rng.uniform(0, 2 * math.pi))
            k_cm = FourVector.from_p3_mass(kt * math.cos(phi), kt * math.sin(phi),
                                           pb * (x1 - x2), M)
            use_ue = self.underlying_event and self._attempt < MAX_EVENT_ATTEMPTS // 2
            ue_cm = self._underlying_event() if use_ue else []
            rest_cm = FourVector.sum([FourVector(beam.sqrt_s, 0.0, 0.0, 0.0), -k_cm,
                                      *(-v for _, v in ue_cm)])
            if rest_cm.E <= 0 or rest_cm.mass2 <= (2 * m_p + self.REMNANT_MASS_MARGIN) ** 2:
                continue
            # accepted: move everything to the lab, close the remnants against P_init(lab)
            k_lab = beam.to_lab(k_cm)
            ue_lab = [(pid, beam.to_lab(v)) for pid, v in ue_cm]
            for pid, v in ue_lab:
                b.add(pid, v)
            rest = FourVector.sum([beam.initial_p4, -k_lab, *(-v for _, v in ue_lab)])
            # the two remnants: an exact 2-body split of the leftover system along ±z
            r1, r2 = two_body_decay(rest, rest.mass, m_p, m_p, 1.0, 0.0)
            b.add(REMNANT, r1, mass=m_p, name="p-remnant(+z)", status=Status.REMNANT)
            b.add(REMNANT, r2, mass=m_p, name="p-remnant(-z)", status=Status.REMNANT)
            return k_lab, M
        raise _RetryEvent

    # ---- finalization ----------------------------------------------------------------
    def _finalize(self, b: _EventBuilder, ch: Channel, true_mass: float) -> Event:
        beam = self.beam
        # beam remnants carry whatever electric charge the rest of the final state does not
        rem = [p for p in b.particles if p.status is Status.REMNANT]
        if rem:
            q_other = math.fsum(p.charge for p in b.particles
                                if p.is_final and p.status is not Status.REMNANT)
            q = round(beam.initial_charge - q_other)
            rem[0].charge, rem[1].charge = float(q - q // 2), float(q // 2)
        ev = Event(
            event_id=self._next_id,
            channel=ch.key,
            beam=beam,
            initial_p4=beam.initial_p4,
            initial_charge=beam.initial_charge,
            particles=b.particles,
            primary_vertex=b.primary_vertex,
            true_mass=true_mass,
            is_signal=ch.kind == "signal",
        )
        self._next_id += 1
        return ev


# --------------------------------------------------------------------------------------
# Channel implementations
# --------------------------------------------------------------------------------------


def _resonance_to_pair(
    gen: CollisionGenerator, b: _EventBuilder, res: int, d1: int, d2: int, angular: str
) -> float:
    m_thr = pdg(d1).mass + pdg(d2).mass
    M = gen.bw_mass(res, m_thr, gen.max_system_mass, require_peak=True)
    k, M = gen.produce(b, M)
    i = b.add(res, k, mass=M)
    b.decay_2body(i, d1, d2, angular=angular)
    return M


def _pick_flavour(gen: CollisionGenerator, flavour: str | None, allowed=("e", "mu")) -> int:
    if flavour in (None, "random"):
        flavour = allowed[int(gen.rng.integers(len(allowed)))]
    if flavour not in LEPTON_IDS:
        raise ValueError(f"unknown lepton flavour {flavour!r}")
    return LEPTON_IDS[flavour]


_FOUR_LEPTON = {"2mu2e": (MUON, ELECTRON), "2e2mu": (ELECTRON, MUON),
                "4mu": (MUON, MUON), "4e": (ELECTRON, ELECTRON)}


def _four_lepton_flavours(gen: CollisionGenerator, final_state: str) -> tuple[int, int]:
    if final_state == "random":
        final_state = ("2mu2e", "2e2mu", "4mu", "4e")[
            int(gen.rng.choice(4, p=[0.25, 0.25, 0.25, 0.25]))
        ]
    if final_state not in _FOUR_LEPTON:
        raise ValueError(f"final_state must be one of {sorted(_FOUR_LEPTON)} or 'random'")
    return _FOUR_LEPTON[final_state]


def _make_dilepton_resonance(res: int, lep: int):
    return lambda gen, b: _resonance_to_pair(gen, b, res, lep, -lep, "1+cos2")


for _lep, _tag in ((MUON, "mumu"), (ELECTRON, "ee")):
    _sym = "μ⁺μ⁻" if _lep == MUON else "e⁺e⁻"
    _channel(f"jpsi_{_tag}", f"J/ψ → {_sym}", "signal", J_PSI,
             "Charmonium resonance — the 1974 November Revolution.",
             {"ee": 3.1, "pp": 1000.0}, f"dy_{_tag}", (2.6, 3.6))(
        _make_dilepton_resonance(J_PSI, _lep))
    _channel(f"z_{_tag}", f"Z⁰ → {_sym}", "signal", Z0,
             "Neutral weak boson — LEP's precision workhorse.",
             {"ee": 91.19, "pp": 1000.0}, f"dy_{_tag}", (60.0, 120.0))(
        _make_dilepton_resonance(Z0, _lep))

_channel("z_tautau", "Z⁰ → τ⁺τ⁻", "signal", Z0,
         "τ leptons decay inside the beam pipe (cτ ≈ 87 µm) into leptons/hadrons + neutrinos.",
         {"ee": 91.19, "pp": 1000.0}, "dy_tautau", (60.0, 120.0))(
    _make_dilepton_resonance(Z0, TAU))


@_channel("h_gammagamma", "H⁰ → γγ", "signal", HIGGS,
          "Diphoton Higgs discovery channel (2012).", {"ee": 126.0, "pp": 1000.0},
          "diphoton", (100.0, 160.0))
def _h_gammagamma(gen: CollisionGenerator, b: _EventBuilder) -> float:
    return _resonance_to_pair(gen, b, HIGGS, PHOTON, PHOTON, "isotropic")


@_channel("h_zz_4l", "H⁰ → ZZ* → 4ℓ", "signal", HIGGS,
          "The 'golden' four-lepton Higgs channel; one Z is off-shell (Z*).",
          {"ee": 126.0, "pp": 1000.0}, "zz_4l", (80.0, 170.0))
def _h_zz_4l(gen: CollisionGenerator, b: _EventBuilder, final_state: str = "random") -> float:
    la, lb = _four_lepton_flavours(gen, final_state)
    ma, mb = pdg(la).mass, pdg(lb).mass
    m_h = gen.bw_mass(HIGGS, 2 * ma + 2 * mb, gen.max_system_mass, require_peak=True)
    k, m_h = gen.produce(b, m_h)
    h = b.add(HIGGS, k, mass=m_h)
    m_z1 = gen.bw_mass(Z0, 2 * ma, m_h - 2 * mb)  # (nearly) on-shell Z
    m_z2 = gen.bw_mass(Z0, 2 * mb, m_h - m_z1)  # off-shell Z*
    z1, z2 = two_body_decay(k, m_h, m_z1, m_z2, sample_cos_theta(gen.rng),
                            gen.rng.uniform(0, 2 * math.pi))
    iz1, = b.split(h, [(Z0, z1, m_z1)])
    iz2, = b.split(h, [(Z0, z2, m_z2)], name="Z0*")
    b.decay_2body(iz1, la, -la)
    b.decay_2body(iz2, lb, -lb)
    return m_h


@_channel("ww_lnulnu", "W⁺W⁻ → ℓ⁺ν ℓ⁻ν̄", "signal", W_PLUS,
          "Boson-pair production; two neutrinos give large missing transverse momentum.",
          {"ee": 200.0, "pp": 1000.0})
def _ww(gen: CollisionGenerator, b: _EventBuilder, flavours: tuple[str, str] | None = None):
    f1, f2 = flavours or (None, None)
    l1, l2 = _pick_flavour(gen, f1), _pick_flavour(gen, f2)  # W+ → l1+ ν,  W- → l2- ν̄
    m1, m2 = pdg(l1).mass, pdg(l2).mass
    m_max = gen.max_system_mass
    m_wp = gen.bw_mass(W_PLUS, m1, m_max - m2)
    m_wm = gen.bw_mass(W_PLUS, m2, m_max - m_wp)
    m_ww = gen.continuum_mass(m_wp + m_wm, m_max, power=4.0)
    k, m_ww = gen.produce(b, m_ww)
    wp, wm = two_body_decay(k, m_ww, m_wp, m_wm, sample_cos_theta(gen.rng),
                            gen.rng.uniform(0, 2 * math.pi))
    iwp = b.add(W_PLUS, wp, mass=m_wp)
    iwm = b.add(-W_PLUS, wm, mass=m_wm)
    b.decay_2body(iwp, -l1, l1 + 1)  # ℓ⁺ ν_ℓ   (ν id = ℓ id + 1)
    b.decay_2body(iwm, l2, -(l2 + 1))  # ℓ⁻ ν̄_ℓ
    return m_ww


# ---- continuum backgrounds -----------------------------------------------------------


def _make_drell_yan(lep: int):
    def _dy(gen: CollisionGenerator, b: _EventBuilder,
            mass_window: tuple[float, float] | None = None) -> float:
        lo, hi = gen.default_window(2 * pdg(lep).mass, mass_window)
        M = gen.continuum_mass(lo, hi, power=3.0)
        k, M = gen.produce(b, M)
        g = b.add(PHOTON, k, mass=M, name="gamma*/Z (DY)", ctau_mm=0.0)
        b.decay_2body(g, lep, -lep, angular="1+cos2")
        return M

    return _dy


for _lep, _tag, _sym in ((MUON, "mumu", "μ⁺μ⁻"), (ELECTRON, "ee", "e⁺e⁻"), (TAU, "tautau", "τ⁺τ⁻")):
    _channel(f"dy_{_tag}", f"γ*/Z → {_sym} continuum", "background", None,
             "Drell–Yan (pp) / QED s-channel (e⁺e⁻) non-resonant lepton pairs.",
             {"ee": 91.19, "pp": 1000.0})(_make_drell_yan(_lep))


@_channel("diphoton", "γγ continuum", "background", None,
          "Non-resonant diphoton production (gg/qq̄ → γγ, e⁺e⁻ → γγ).",
          {"ee": 126.0, "pp": 1000.0})
def _diphoton(gen: CollisionGenerator, b: _EventBuilder,
              mass_window: tuple[float, float] | None = None) -> float:
    lo, hi = gen.default_window(0.0, mass_window)
    M = gen.continuum_mass(lo, hi, power=3.0)
    k, M = gen.produce(b, M)
    g1, g2 = two_body_decay(k, M, 0.0, 0.0, sample_cos_theta(gen.rng, "1+cos2"),
                            gen.rng.uniform(0, 2 * math.pi))
    b.add(PHOTON, g1, mass=0.0)
    b.add(PHOTON, g2, mass=0.0)
    return M


@_channel("zz_4l", "ZZ(*) → 4ℓ continuum", "background", None,
          "Non-resonant Z/γ* pair production — the irreducible H → 4ℓ background.",
          {"ee": 250.0, "pp": 1000.0})
def _zz_4l(gen: CollisionGenerator, b: _EventBuilder,
           mass_window: tuple[float, float] | None = None, final_state: str = "random") -> float:
    la, lb = _four_lepton_flavours(gen, final_state)
    ma, mb = pdg(la).mass, pdg(lb).mass
    lo, hi = gen.default_window(2 * ma + 2 * mb, mass_window)
    m_sys = gen.continuum_mass(lo, hi, power=4.0)
    m_z1 = gen.bw_mass(Z0, 2 * ma, m_sys - 2 * mb)
    m_z2 = gen.bw_mass(Z0, 2 * mb, m_sys - m_z1)
    k, m_sys = gen.produce(b, m_sys)
    z1, z2 = two_body_decay(k, m_sys, m_z1, m_z2, sample_cos_theta(gen.rng),
                            gen.rng.uniform(0, 2 * math.pi))
    iz1 = b.add(Z0, z1, mass=m_z1)
    iz2 = b.add(Z0, z2, mass=m_z2)
    b.decay_2body(iz1, la, -la)
    b.decay_2body(iz2, lb, -lb)
    return m_sys


# --------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------

CONSERVATION_TOLERANCE_GEV = 1e-12
ON_SHELL_REL_TOLERANCE = 1e-9  # |m² − m₀²| / E²


def _on_shell_violation(ev: Event) -> float:
    worst = 0.0
    for p in ev.final_state():
        e2 = max(p.p4.E ** 2, 1.0)
        worst = max(worst, abs(p.p4.mass2 - p.mass**2) / e2)
    return worst


def _self_test_matrix() -> list[tuple[BeamConfig, list[str]]]:
    lepton_bkg = ["dy_mumu", "dy_ee", "dy_tautau"]
    return [
        (BeamConfig("ee", 3.1), ["jpsi_mumu", "jpsi_ee", "dy_mumu", "dy_ee"]),
        (BeamConfig.from_beam_energies("ee", 7.0, 4.0),  # Belle II-like asymmetric
         ["jpsi_mumu", "jpsi_ee", *lepton_bkg, "diphoton"]),
        (BeamConfig("ee", 91.19), ["z_mumu", "z_ee", "z_tautau", *lepton_bkg, "diphoton"]),
        (BeamConfig("ee", 250.0), ["h_gammagamma", "h_zz_4l", "ww_lnulnu", "zz_4l",
                                   "z_mumu", "diphoton"]),
        (BeamConfig("pp", 1000.0), sorted(CHANNELS)),
        (BeamConfig("pp", 300.0, beta_z=0.4), sorted(CHANNELS)),
        (BeamConfig("pp", 13.0), ["jpsi_mumu", "jpsi_ee", "dy_mumu"]),
    ]


def run_self_test(n_events: int = 200, seed: int = 20260704) -> bool:
    print("=" * 96)
    print("POCKET CERN COLLIDER - physics_engine self-test")
    print("=" * 96)
    print("\nPDG catalogue (Scikit-HEP `particle`, natural units):")
    print(f"  {'id':>6} {'name':<12}{'mass [GeV]':>14}{'width [GeV]':>14}{'Q':>5}"
          f"{'ctau [mm]':>13}{'J':>5}  kind")
    for pid, s in pdg_catalog().items():
        ct = "stable" if s.is_stable else f"{s.ctau_mm:.4g}"
        print(f"  {pid:>6} {s.name:<12}{s.mass:14.7g}{s.width:14.5g}{s.charge:+5.0f}"
              f"{ct:>13}{s.spin if s.spin is not None else '-':>5}  {s.kind}")

    # Lorentz-boost sanity: Λᵀ g Λ = g for a random ultra-relativistic boost
    rng = np.random.default_rng(seed)
    bv = rng.normal(size=3)
    bv *= 0.999999 / np.linalg.norm(bv)
    lam = lorentz_boost_matrix(bv)
    metric_err = float(np.max(np.abs(lam.T @ MINKOWSKI_METRIC @ lam - MINKOWSKI_METRIC)))
    print(f"\nLorentz group check  max|Λᵀ g Λ − g| (|β| = 0.999999): {metric_err:.2e}")

    print(f"\n4-momentum & charge conservation, {n_events} events per row "
          f"(tolerance {CONSERVATION_TOLERANCE_GEV:.0e} GeV):")
    head = (f"  {'beams':<42}{'channel':<14}{'<M_true>':>10}{'max|dP|':>11}"
            f"{'max|dQ|':>9}{'on-shell':>10}{'<MET>':>9}  ok")
    print(head)
    print("  " + "-" * (len(head) - 2))
    all_ok = True
    for i, (beam, channels) in enumerate(_self_test_matrix()):
        gen = CollisionGenerator(beam, seed=seed + i)
        for ch in channels:
            evs = gen.generate_many(ch, n_events)
            dp = max(e.max_conservation_violation() for e in evs)
            dq = max(abs(e.charge_residual()) for e in evs)
            osv = max(_on_shell_violation(e) for e in evs)
            met = float(np.mean([e.truth_missing_pt()[2] for e in evs]))
            mt = float(np.mean([e.true_mass for e in evs]))
            ok = dp < CONSERVATION_TOLERANCE_GEV and dq == 0 and osv < ON_SHELL_REL_TOLERANCE
            all_ok &= ok
            print(f"  {beam.label():<42}{ch:<14}{mt:10.3f}{dp:11.1e}{dq:9.0f}"
                  f"{osv:10.1e}{met:9.2f}  {'PASS' if ok else 'FAIL'}")
    metric_ok = metric_err < 1e-9
    all_ok &= metric_ok

    print("\nSample event: H -> ZZ* -> mu+ mu- e+ e-  (pp, sqrt(s) = 1000 GeV)")
    h_ev = CollisionGenerator(BeamConfig("pp", 1000.0), seed=125).generate(
        "h_zz_4l", final_state="2mu2e")
    print(h_ev.table())
    leptons = [p for p in h_ev.final_state() if abs(p.pdg_id) in (ELECTRON, MUON)]
    m4l = FourVector.sum(p.p4 for p in leptons).mass
    print(f"Reconstructed m(4l) from truth leptons = {m4l:.6f} GeV "
          f"(generated M_H = {h_ev.true_mass:.6f} GeV)")

    print("\nSample event: Z0 -> mu+ mu-  (e+e-, sqrt(s) = 91.19 GeV)")
    z_ev = CollisionGenerator(BeamConfig("ee", 91.19), seed=91).generate("z_mumu")
    print(z_ev.table())
    mus = [p for p in z_ev.final_state() if abs(p.pdg_id) == MUON]
    print(f"Reconstructed m(mumu) = {FourVector.sum(p.p4 for p in mus).mass:.6f} GeV")

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SELF-TEST FAILED"))
    return all_ok


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    sys.exit(0 if run_self_test(n) else 1)
