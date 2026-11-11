"""Milestone 1 verification: exact relativistic kinematics & conservation laws."""

import math

import numpy as np
import pytest

import physics_engine as pe
from physics_engine import (
    CHANNELS,
    MINKOWSKI_METRIC,
    BeamConfig,
    CollisionGenerator,
    FourVector,
    KinematicsError,
    lorentz_boost_matrix,
    pdg,
)

TOL = pe.CONSERVATION_TOLERANCE_GEV


# ---- PDG database -------------------------------------------------------------------


def test_pdg_lookup_natural_units():
    z = pdg(pe.Z0)
    assert z.mass == pytest.approx(91.19, abs=0.01)  # GeV, not MeV
    assert z.width == pytest.approx(2.495, abs=0.01)
    assert pdg(pe.W_PLUS).charge == 1 and pdg(-pe.W_PLUS).charge == -1
    assert pdg(pe.HIGGS).mass == pytest.approx(125.2, abs=0.2)
    assert pdg(pe.J_PSI).mass == pytest.approx(3.0969, abs=1e-3)
    assert pdg(pe.TAU).ctau_mm == pytest.approx(0.0870, rel=0.01)
    assert pdg(pe.NU_MU).mass == 0.0 and pdg(pe.NU_MU).is_neutrino
    assert pdg(pe.PROTON).is_stable and pdg(pe.ELECTRON).spin == 0.5
    for pid in pe.KEY_PARTICLES:
        assert pdg(pid).mass >= 0.0


# ---- Lorentz algebra ----------------------------------------------------------------


@pytest.mark.parametrize("beta_mag", [0.0, 1e-8, 0.3, 0.9, 0.999999])
def test_boost_matrix_preserves_metric(beta_mag):
    rng = np.random.default_rng(1)
    b = rng.normal(size=3)
    b *= beta_mag / np.linalg.norm(b)
    lam = lorentz_boost_matrix(b)
    gamma = 1 / math.sqrt(1 - beta_mag**2)
    assert np.allclose(lam.T @ MINKOWSKI_METRIC @ lam, MINKOWSKI_METRIC, atol=1e-12 * gamma**2)
    # Λ(β) Λ(−β) = 1
    assert np.allclose(lam @ lorentz_boost_matrix(-b), np.eye(4), atol=1e-12 * gamma**2)


def test_boost_of_particle_at_rest():
    m, p = 91.1876, np.array([10.0, -20.0, 30.0])
    e = math.sqrt(m * m + p @ p)
    moving = FourVector(m, 0, 0, 0).boost(p / e, e / m)
    assert moving.E == pytest.approx(e, rel=1e-14)
    assert np.allclose(moving.p3, p, rtol=1e-13)
    assert moving.mass == pytest.approx(m, rel=1e-12)


def test_kallen_momentum_pion_decay():
    # π⁺ → μ⁺ ν_μ: p* = 29.79 MeV
    p = pe.kallen_momentum(pdg(pe.PI_PLUS).mass, pdg(pe.MUON).mass, 0.0)
    assert p == pytest.approx(0.02979, abs=1e-5)
    with pytest.raises(KinematicsError):
        pe.kallen_momentum(1.0, 0.6, 0.6)


def test_two_body_decay_closure_matches_independent_boost():
    M, m1, m2 = 125.0, 91.0, 30.0
    parent = FourVector.from_p3_mass(40.0, -15.0, 600.0, M)
    c, phi = 0.3, 1.1
    d1, d2 = pe.two_body_decay(parent, M, m1, m2, c, phi)
    # daughter 2 from closure must agree with daughter 2 boosted independently
    p = pe.kallen_momentum(M, m1, m2)
    s = math.sqrt(1 - c * c)
    d2_rest = FourVector.from_p3_mass(-p * s * math.cos(phi), -p * s * math.sin(phi), -p * c, m2)
    d2_boost = d2_rest.boost(parent.beta_vec, parent.E / M)
    assert np.allclose(d2.to_array(), d2_boost.to_array(), atol=1e-10)
    assert d1.mass == pytest.approx(m1, rel=1e-10)
    assert d2.mass == pytest.approx(m2, rel=1e-10)
    assert (d1 + d2).mass == pytest.approx(M, rel=1e-12)


def test_three_body_phase_space_conserves():
    rng = np.random.default_rng(3)
    tau = pdg(pe.TAU).mass
    parent = FourVector.from_p3_mass(1.0, 2.0, 45.0, tau)
    for _ in range(200):
        vs = pe.three_body_decay(rng, parent, tau, (pdg(pe.MUON).mass, 0.0, 0.0))
        res = (parent - FourVector.sum(vs)).to_array()
        assert np.max(np.abs(res)) < 1e-13


# ---- Breit–Wigner -------------------------------------------------------------------


def test_breit_wigner_truncation_and_shape():
    rng = np.random.default_rng(11)
    z = pdg(pe.Z0)
    ms = np.array([pe.sample_breit_wigner_mass(rng, z.mass, z.width, 80.0, 100.0)
                   for _ in range(20000)])
    assert ms.min() >= 80.0 and ms.max() <= 100.0
    assert np.median(ms) == pytest.approx(z.mass, abs=0.1)
    # half-width at half maximum ≈ Γ/2
    inside = np.mean(np.abs(ms - z.mass) < z.width / 2)
    assert inside == pytest.approx(0.5, abs=0.05)  # ~ (2/π)atan(1) of the (truncated) BW
    with pytest.raises(KinematicsError):
        pe.sample_breit_wigner_mass(rng, z.mass, z.width, 50.0, 40.0)


def test_breit_wigner_pdf_normalised():
    z = pdg(pe.Z0)
    m = np.linspace(0.0, 2000.0, 2_000_001)
    integral = np.trapezoid(pe.relativistic_breit_wigner_pdf(m, z.mass, z.width), m)
    assert integral == pytest.approx(1.0, abs=1e-4)
    # narrow resonance too (J/ψ, Γ/m ~ 3e-5)
    j = pdg(pe.J_PSI)
    m = np.linspace(j.mass - 2.0, j.mass + 2.0, 4_000_001)
    integral = np.trapezoid(pe.relativistic_breit_wigner_pdf(m, j.mass, j.width), m)
    assert integral == pytest.approx(1.0, abs=1e-4)


# ---- Beams --------------------------------------------------------------------------


def test_asymmetric_beams_preserve_sqrt_s():
    beam = BeamConfig.from_beam_energies("ee", 7.0, 4.0)
    assert beam.sqrt_s == pytest.approx(2 * math.sqrt(7.0 * 4.0), rel=1e-6)
    assert beam.initial_p4.mass == pytest.approx(beam.sqrt_s, rel=1e-12)
    assert beam.initial_p4.E == pytest.approx(11.0, rel=1e-9)
    assert beam.initial_p4.pz == pytest.approx(3.0, rel=1e-6)


@pytest.mark.parametrize("bad", [dict(kind="mumu"), dict(sqrt_s=1.0), dict(sqrt_s=5000.0),
                                 dict(beta_z=1.0)])
def test_beam_validation(bad):
    with pytest.raises(ValueError):
        BeamConfig(**{"kind": "ee", "sqrt_s": 100.0, **bad})


def test_below_threshold_raises():
    with pytest.raises(KinematicsError):
        CollisionGenerator(BeamConfig("ee", 50.0), seed=0).generate("z_mumu")
    with pytest.raises(KinematicsError):
        CollisionGenerator(BeamConfig("ee", 100.0), seed=0).generate("h_gammagamma")


# ---- Full events: conservation ------------------------------------------------------

_CONFIGS = [
    BeamConfig("ee", 91.19),
    BeamConfig("ee", 250.0),
    BeamConfig("pp", 1000.0),
    BeamConfig("pp", 500.0, beta_z=-0.6),
]


def _reachable(beam: BeamConfig):
    gen = CollisionGenerator(beam, seed=0)
    out = []
    for key in sorted(CHANNELS):
        try:
            gen.generate(key)
        except KinematicsError:
            continue
        out.append(key)
    return out


@pytest.mark.parametrize("beam", _CONFIGS, ids=lambda b: b.label())
def test_conservation_all_channels(beam):
    gen = CollisionGenerator(beam, seed=42)
    channels = _reachable(beam)
    assert "z_mumu" in channels
    for key in channels:
        for ev in gen.generate_many(key, 60):
            assert ev.max_conservation_violation() < TOL, (key, ev.conservation_residual())
            assert ev.charge_residual() == 0
            assert pe._on_shell_violation(ev) < pe.ON_SHELL_REL_TOLERANCE
            # tree consistency: every decayed particle's daughters sum to it
            for p in ev.particles:
                if p.daughters:
                    s = FourVector.sum(ev.particles[d].p4 for d in p.daughters)
                    assert np.max(np.abs((s - p.p4).to_array())) < TOL


def test_h_to_4l_cascade_structure():
    ev = CollisionGenerator(BeamConfig("pp", 1000.0), seed=5).generate(
        "h_zz_4l", final_state="2mu2e")
    (h,) = ev.find(pe.HIGGS)
    zs = [ev.particles[i] for i in h.daughters]
    assert [z.pdg_id for z in zs] == [pe.Z0, pe.Z0]
    leptons = [ev.particles[i] for z in zs for i in z.daughters]
    assert sorted(p.pdg_id for p in leptons) == [-13, -11, 11, 13]
    m4l = FourVector.sum(p.p4 for p in leptons).mass
    assert m4l == pytest.approx(ev.true_mass, rel=1e-12)


def test_ww_has_missing_pt_and_tau_flies():
    gen = CollisionGenerator(BeamConfig("ee", 200.0), seed=8)
    mets = [gen.generate("ww_lnulnu").truth_missing_pt()[2] for _ in range(100)]
    assert np.mean(mets) > 20.0

    # τ flight length follows βγcτ: <L/(βγ)> → cτ
    gen = CollisionGenerator(BeamConfig("ee", 91.19), seed=9)
    ratios = []
    for _ in range(300):
        for t in gen.generate("z_tautau").find(pe.TAU, -pe.TAU):
            assert t.decay_vertex is not None
            ratios.append(t.flight_length_mm / (t.p4.p / t.mass))
    assert np.mean(ratios) == pytest.approx(pdg(pe.TAU).ctau_mm, rel=0.1)


def test_daughter_vertices_start_at_mother_decay_vertex():
    ev = CollisionGenerator(BeamConfig("ee", 91.19), seed=4).generate("z_tautau")
    for p in ev.particles:
        if p.mother is not None:
            assert p.production_vertex == ev.particles[p.mother].decay_vertex


def test_reproducible_with_seed():
    a = CollisionGenerator(BeamConfig("pp", 1000.0), seed=99).generate("z_mumu")
    b = CollisionGenerator(BeamConfig("pp", 1000.0), seed=99).generate("z_mumu")
    assert [p.p4 for p in a.particles] == [p.p4 for p in b.particles]
