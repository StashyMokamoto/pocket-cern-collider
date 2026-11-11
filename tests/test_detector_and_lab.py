"""Milestone 2 verification: detector simulation + invariant-mass analysis lab."""

import math

import numpy as np
import pytest

import detector_sim as ds
import invariance_lab as il
import physics_engine as pe
from detector_sim import Detector, DetectorConfig, Helix, Layer
from physics_engine import BeamConfig, CollisionGenerator, FourVector


# ---- particle gun -------------------------------------------------------------------


def gun(*particles, beam=BeamConfig("ee", 91.19)) -> pe.Event:
    """Hand-built truth event: each particle is (pdg_id, pt, eta, phi[, vertex_mm])."""
    parts = []
    for i, spec in enumerate(particles):
        pid, pt, eta, phi, *rest = spec
        vtx = rest[0] if rest else (0.0, 0.0, 0.0)
        s = pe.pdg(pid)
        parts.append(pe.GenParticle(i, pid, s.name, FourVector.from_pt_eta_phi_m(pt, eta, phi, s.mass),
                                    s.mass, s.charge, pe.Status.FINAL, None,
                                    production_vertex=vtx))
    return pe.Event(0, "gun", beam, beam.initial_p4, 0.0, parts, (0.0, 0.0, 0.0), 0.0, False)


def sim(*particles, config=None, seed=0):
    return Detector(config, seed=seed).simulate(gun(*particles))


def layer_energy(trace, layer):
    return sum(d.energy for d in trace.deposits if d.layer is layer)


# ---- stopping & visibility rules ----------------------------------------------------


def test_neutrino_is_completely_invisible():
    d = sim((pe.NU_MU, 40.0, 0.3, 1.0))
    (t,) = d.traces
    assert t.fate == "invisible" and not t.visible_track and not t.deposits
    assert d.objects == [] and d.layer_summary()["tracker_hits"] == 0


def test_photon_no_track_stops_in_ecal_at_arrival_phi():
    d = sim((pe.PHOTON, 50.0, 0.5, 2.0))
    (t,) = d.traces
    assert t.fate == "stopped_ecal" and t.track is None and not t.visible_track
    assert layer_energy(t, Layer.HCAL) == 0 and layer_energy(t, Layer.ECAL) > 0
    dep = t.deposits[0]
    assert math.hypot(*dep.position_m[:2]) == pytest.approx(ds.R_ECAL[0])
    assert dep.phi == pytest.approx(2.0, abs=1e-9)
    assert [o.kind for o in d.objects] == [ds.PHOTON]


def test_electron_track_and_ecal_deposit_at_helix_impact():
    d = sim((pe.ELECTRON, 5.0, 0.0, 0.0))
    (t,) = d.traces
    assert t.fate == "stopped_ecal" and t.visible_track and len(t.track.hits) == 13
    assert layer_energy(t, Layer.HCAL) == 0
    dep = t.deposits[0]
    assert math.hypot(*dep.position_m[:2]) == pytest.approx(ds.R_ECAL[0], rel=1e-9)
    # the impact azimuth is bent away from the initial φ = 0 (not the momentum direction)
    R = 5.0 / (ds.LARMOR_CONSTANT * 3.8)
    expected = 2 * math.asin(ds.R_ECAL[0] / (2 * R))  # chord angle, e⁻ bends counter-clockwise
    assert dep.phi == pytest.approx(+0.5 * expected, rel=1e-6)
    assert d.objects[0].kind == ds.ELECTRON and d.objects[0].charge == -1


def test_charged_hadron_mip_in_ecal_shower_in_hcal():
    d = sim((pe.PI_PLUS, 20.0, 0.2, 1.0))
    (t,) = d.traces
    assert t.fate == "stopped_hcal" and t.visible_track
    e_ecal, e_hcal = layer_energy(t, Layer.ECAL), layer_energy(t, Layer.HCAL)
    true_e = d.truth.particles[0].p4.E
    assert 0 < e_ecal < 0.2 * true_e and e_hcal > 0.5 * true_e
    assert d.objects[0].kind == ds.CHARGED_HADRON


def test_neutral_hadron_no_track_hcal_only():
    d = sim((pe.NEUTRON, 30.0, -0.4, -2.0))
    (t,) = d.traces
    assert t.fate == "stopped_hcal" and t.track is None
    assert layer_energy(t, Layer.ECAL) == 0 and layer_energy(t, Layer.HCAL) > 0
    assert d.objects[0].kind == ds.NEUTRAL_HADRON


def test_muon_punches_through_to_muon_system():
    d = sim((pe.MUON, 30.0, 0.1, 0.5))
    (t,) = d.traces
    assert t.fate == "exited_muon" and Layer.MUON in t.layers_reached
    assert len(t.track.muon_hits) == len(ds.MUON_STATIONS_M)
    assert 0 < layer_energy(t, Layer.ECAL) < 2 and 0 < layer_energy(t, Layer.HCAL) < 6
    end = t.path[-1]
    assert math.hypot(end[0], end[1]) == pytest.approx(ds.R_MUON[1], rel=1e-9)
    assert d.objects[0].kind == ds.MUON


def test_forward_particles_escape_unmeasured():
    d = sim((pe.MUON, 30.0, 2.7, 0.0), (pe.PHOTON, 30.0, -3.0, 0.0))
    assert all(t.fate == "escaped_forward" for t in d.traces) and not d.objects


# ---- Larmor helix --------------------------------------------------------------------


def test_larmor_radius_formula():
    h = Helix(0, 0, 0, pt=1.0, phi0=0.0, cot_theta=0.0, charge=1, b_field_t=3.8)
    assert h.radius == pytest.approx(1.0 / (0.299792458 * 3.8))  # ≈ 0.878 m


@pytest.mark.parametrize("charge,b,expected_sign", [(+1, 3.8, -1), (-1, 3.8, +1),
                                                    (+1, -3.8, +1), (-1, -3.8, -1)])
def test_curvature_sign(charge, b, expected_sign):
    """F = q v × B: a μ⁺ launched along +x in B_z > 0 bends towards −y (clockwise)."""
    h = Helix(0, 0, 0, pt=5.0, phi0=0.0, cot_theta=0.0, charge=charge, b_field_t=b)
    x, y, _ = h.point_at(0.5)
    assert x > 0 and np.sign(y) == expected_sign
    # the same sign is visible in the simulated muon track
    d = sim((pe.MUON if charge < 0 else -pe.MUON, 5.0, 0.0, 0.0),
            config=DetectorConfig(b_field_t=b))
    assert np.sign(d.traces[0].path[-1][1]) == expected_sign


def test_zero_field_gives_straight_tracks():
    d = sim((pe.MUON, 2.0, 0.3, 0.8), config=DetectorConfig(b_field_t=0.0))
    path = d.traces[0].path
    phis = np.arctan2(path[1:, 1], path[1:, 0])
    assert np.allclose(phis, 0.8, atol=1e-12)
    assert d.traces[0].track.radius_m == math.inf


def test_helix_crossing_is_exact_and_loopers_are_safe():
    h = Helix(0.0005, -0.0003, 0.0, pt=3.0, phi0=1.2, cot_theta=0.5, charge=-1, b_field_t=3.8)
    for r in ds.SILICON_LAYERS_M:
        s = h.arc_to_radius(r)
        x, y, z = h.point_at(s)
        assert math.hypot(x, y) == pytest.approx(r, abs=1e-12)
        assert z == pytest.approx(0.5 * s)
    # looper: 2R ≈ 0.53 m < ECAL radius → never reaches it, no math-domain error
    loop = Helix(0, 0, 0, pt=0.3, phi0=0.0, cot_theta=0.1, charge=1, b_field_t=3.8)
    assert loop.max_radius == pytest.approx(2 * loop.radius)
    assert loop.arc_to_radius(ds.R_ECAL[0]) is None
    d = sim((pe.PI_PLUS, 0.3, 0.1, 0.0))
    t = d.traces[0]
    assert t.fate == "looper" and not t.deposits
    assert np.max(np.hypot(t.path[:, 0], t.path[:, 1])) <= 2 * loop.radius + 1e-9


def test_displaced_vertex_track_starts_at_vertex():
    vtx_mm = (12.0, -5.0, 3.0)
    d = sim((pe.PI_PLUS, 4.0, 0.2, -0.4, vtx_mm))
    assert np.allclose(d.traces[0].path[0], np.array(vtx_mm) * 1e-3)
    zoomed = ds.exaggerate_displacement(d.traces[0].path[0], (0, 0, 0), 10.0)
    assert np.allclose(zoomed, np.array(vtx_mm) * 1e-2)


# ---- resolution ----------------------------------------------------------------------


def test_resolution_formulas():
    assert ds.resolution_tracker_pt(100.0) == pytest.approx(math.hypot(0.01, 0.05))
    assert ds.resolution_calo(100.0, 0.10, 0.01) == pytest.approx(math.hypot(0.01, 0.01))
    det = Detector(seed=3)
    es = np.array([det.smear_calo(50.0, Layer.ECAL) for _ in range(20000)])
    assert np.std(es) / 50.0 == pytest.approx(ds.resolution_calo(50.0, 0.10, 0.01), rel=0.05)
    pts = np.array([det.smear_track(40.0, 1)[0] for _ in range(20000)])
    assert np.std(pts) / 40.0 == pytest.approx(ds.resolution_tracker_pt(40.0), rel=0.06)


def test_dimuon_mass_reconstruction_accuracy():
    gen = CollisionGenerator(BeamConfig("pp", 1000.0), seed=11)
    det = Detector(seed=12)
    res = []
    for _ in range(300):
        c = il.reconstruct(det.simulate(gen.generate("z_mumu")), "mumu")
        if c:
            assert c.mass == pytest.approx(il.invariant_mass(c.objects))
            res.append(c.residual)
    res = np.array(res)
    assert len(res) > 200
    assert abs(np.median(res)) < 0.3  # unbiased
    assert 0.5 < 1.4826 * np.median(np.abs(res - np.median(res))) < 2.5  # ~1–2 GeV


def test_invariant_mass_formula():
    a = FourVector.from_p3_mass(30.0, 0.0, 10.0, 0.105)
    b = FourVector.from_p3_mass(-30.0, 0.0, 10.0, 0.105)
    expected = math.sqrt((a.E + b.E) ** 2 - 20.0**2)
    assert il.invariant_mass([a, b]) == pytest.approx(expected, rel=1e-14)


# ---- MET -----------------------------------------------------------------------------


def test_met_aligns_with_neutrinos_in_ww():
    gen = CollisionGenerator(BeamConfig("ee", 200.0), seed=21)
    det = Detector(seed=22)
    dphi, ratio = [], []
    for _ in range(200):
        d = det.simulate(gen.generate("ww_lnulnu"))
        mex, mey, met = d.truth.truth_missing_pt()
        if met < 20:
            continue
        diff = (d.met_phi - math.atan2(mey, mex) + math.pi) % (2 * math.pi) - math.pi
        dphi.append(abs(diff))
        ratio.append(d.met / met)
    assert len(dphi) > 100
    assert np.median(dphi) < 0.1
    assert np.median(ratio) == pytest.approx(1.0, abs=0.05)


def test_met_small_without_neutrinos():
    d = sim((pe.MUON, 40.0, 0.1, 0.0), (-pe.MUON, 40.0, -0.1, math.pi))
    assert d.met < 3.0


# ---- luminosity run & fit ------------------------------------------------------------


def test_fit_recovers_injected_peak():
    rng = np.random.default_rng(5)
    h = il.Histogram(60.0, 120.0, 60)
    h.fill(91.0 + rng.standard_cauchy(3000) * 1.25)  # Lorentzian Γ = 2.5
    h.fill(60.0 + rng.exponential(25.0, 3000))
    fit = il.fit_resonance(h, "lorentzian", mass_hint=90.0)
    assert fit.success
    assert fit.mass == pytest.approx(91.0, abs=3 * fit.mass_err)
    assert fit.width == pytest.approx(2.5, abs=4 * fit.width_err)
    assert fit.chi2 / fit.ndf < 2.0


def test_z_plus_drell_yan_500_event_luminosity_run():
    run = il.LuminosityRun("z_mumu", BeamConfig("pp", 1000.0), signal_fraction=0.5,
                           seed=2026)
    run.run(500)
    assert run.n_events == 500 and run.histogram.entries > 300
    fit = run.fit("voigt")
    m_z = pe.pdg(pe.Z0).mass
    assert fit.success
    assert fit.mass == pytest.approx(m_z, abs=max(4 * fit.mass_err, 0.5))
    assert 0.0 < fit.mass_err < 0.6
    assert fit.significance > 10
    assert fit.chi2 / fit.ndf < 2.5
    s_true, b_true = run.truth_counts_in(fit.window)
    assert fit.s_window == pytest.approx(s_true, rel=0.25)


@pytest.mark.parametrize("model", il.FIT_MODELS)
def test_all_fit_models_run(model):
    run = il.LuminosityRun("z_ee", BeamConfig("pp", 1000.0), seed=9)
    run.run(400)
    fit = run.fit(model)
    assert fit.success and 85 < fit.mass < 97


# ---- mystery challenge ---------------------------------------------------------------


def test_mystery_public_view_hides_truth():
    game = il.MysteryChallenge(seed=1)
    me = game.new_event("h_zz_4l")
    view = me.public_view()
    text = repr(view)
    for leak in ("h_zz_4l", "H0", "pdg_id", "Z0", "mu-", "fate"):
        assert leak not in text
    assert view["clues"]["candidate_final_state"] == "4l"


def test_mystery_grading():
    game = il.MysteryChallenge(seed=4)
    me = game.new_event("ww_lnulnu")
    truth = me._truth_signature
    g = game.grade(me, dict(truth), "ww_lnulnu")
    assert g.score == 100 and g.particles_correct and g.channel_correct
    visible = {k: v for k, v in truth.items() if k != "neutrino"}
    g = game.grade(me, visible, "z_mumu")
    assert g.particles_partial and not g.channel_correct and g.score == 25
    me = game.new_event("z_mumu")
    g = game.grade(me, ["mu+", "mu-"], "z_ee")
    assert g.particles_correct and g.parent_correct and g.score == 75
    with pytest.raises(ValueError):
        game.grade(me, ["quark"], "z_mumu")


def test_mystery_solvable_from_clues_alone():
    game = il.MysteryChallenge(seed=77)
    scores = []
    for i in range(18):
        me = game.new_event(il.MYSTERY_POOL[i % len(il.MYSTERY_POOL)])
        scores.append(game.grade(me, *il.deduce_from_clues(me.clues)).score)
    assert np.mean(scores) >= 75
