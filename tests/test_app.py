"""Milestone 3 verification: FastAPI control-room API (TestClient integration tests)."""

import json
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as server
import physics_engine as pe


@pytest.fixture(scope="module")
def client():
    return TestClient(server.app)


def _strict_json(resp):
    """Payloads must be strict JSON (no NaN / Infinity tokens)."""
    return json.loads(resp.text, parse_constant=lambda c: pytest.fail(f"non-finite JSON {c}"))


# ---- static page & config -----------------------------------------------------------


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "<canvas id=\"display\"" in r.text and "/api/luminosity/step" in r.text


def test_config(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    cfg = _strict_json(r)
    assert {b["id"] for b in cfg["beams"]} == {"ee", "pp"}
    assert cfg["sqrt_s_range"] == [2.0, 1000.0]
    ch = {c["key"]: c for c in cfg["channels"]}
    for key in ("z_mumu", "z_ee", "h_gammagamma", "h_zz_4l", "ww_lnulnu", "jpsi_mumu"):
        assert ch[key]["kind"] == "signal" and ch[key]["label"]
    assert ch["z_mumu"]["resonance"]["mass"] == pytest.approx(pe.pdg(pe.Z0).mass, abs=1e-6)
    assert ch["h_zz_4l"]["resonance"]["mass"] == pytest.approx(pe.pdg(pe.HIGGS).mass, abs=1e-6)
    # pp thresholds include the two surviving proton remnants
    assert ch["z_mumu"]["min_sqrt_s"]["pp"] - ch["z_mumu"]["min_sqrt_s"]["ee"] == \
        pytest.approx(2 * pe.pdg(pe.PROTON).mass + 0.1, abs=1e-5)
    assert ch["z_mumu"]["luminosity"] and not ch["ww_lnulnu"]["luminosity"]
    assert ch["dy_mumu"]["kind"] == "background"
    layers = {l["id"]: (l["r_min"], l["r_max"]) for l in cfg["detector"]["layers"]}
    assert layers == {"beam_pipe": (0.0, 0.03), "tracker": (0.03, 1.10), "ecal": (1.10, 1.70),
                      "hcal": (1.70, 2.80), "muon": (2.80, 4.50)}
    assert cfg["detector"]["b_field_default"] == 3.8 and cfg["detector"]["eta_max"] == 2.5
    assert len(cfg["mystery"]["pool"]) == 6


# ---- /api/collide -------------------------------------------------------------------


def test_collide_z_mumu_full_payload(client):
    r = client.post("/api/collide", json={"beam": "pp", "channel": "z_mumu", "sqrt_s": 1000,
                                          "b_field_tesla": 3.8, "seed": 11})
    assert r.status_code == 200
    ev = _strict_json(r)
    for key in ("traces", "deposits", "towers", "objects", "met", "candidate", "vertices",
                "balance", "truth", "beam", "primary_vertex_mm"):
        assert key in ev
    assert ev["truth"]["channel"] == "z_mumu" and ev["truth"]["is_signal"]
    assert len(ev["towers"]["ecal"]) == 72 and len(ev["towers"]["hcal"]) == 36
    # polylines are (x, y, z) triples in metres, inside the detector envelope
    muons = [t for t in ev["traces"] if t["class"] == "muon" and t["origin"] == "hard"]
    assert len(muons) == 2
    for t in muons:
        pts = np.array(t["path"])
        assert pts.shape[1] == 3 and len(pts) > 10
        assert np.hypot(pts[:, 0], pts[:, 1]).max() <= 4.5 + 1e-3
        if t["fate"] == "exited_muon":
            assert len(t["muon_hits"]) == 4
    # reconstructed candidate vs MC truth, and exact conservation in the truth record
    c = ev["candidate"]
    assert c["final_state"] == "mumu" and len(c["objects"]) == 2
    assert abs(c["mass"] - c["truth_mass"]) < 6.0
    assert ev["balance"]["truth_max_dP"] < 1e-12
    assert ev["balance"]["truth_charge_residual"] == 0
    # particle-flow MET closes exactly against the visible objects
    assert ev["balance"]["closure"] < 1e-9
    assert ev["met"]["met"] == pytest.approx(math.hypot(ev["met"]["mex"], ev["met"]["mey"]), abs=1e-5)


def test_collide_is_reproducible_with_seed(client):
    body = {"beam": "ee", "channel": "z_ee", "sqrt_s": 91.19, "seed": 5}
    a, b = client.post("/api/collide", json=body).json(), client.post("/api/collide", json=body).json()
    assert a["objects"] == b["objects"] and a["traces"] == b["traces"]


def test_collide_zero_field_gives_straight_tracks(client):
    ev = client.post("/api/collide", json={"beam": "pp", "channel": "h_zz_4l", "sqrt_s": 1000,
                                           "b_field_tesla": 0.0, "seed": 3}).json()
    assert ev["b_field_t"] == 0.0
    charged = [t for t in ev["traces"] if t["charged"] and len(t["path"]) > 1]
    assert charged
    for t in charged:
        p = np.array(t["path"])[:, :2]
        d = p[-1] - p[0]
        # perpendicular distance of every point to the chord ≈ 0 (rounding of 0.1 mm)
        perp = np.abs(d[0] * (p[:, 1] - p[0, 1]) - d[1] * (p[:, 0] - p[0, 0])) / np.hypot(*d)
        assert perp.max() < 2e-4


def test_collide_tau_has_displaced_vertices(client):
    ev = client.post("/api/collide", json={"beam": "ee", "channel": "z_tautau", "sqrt_s": 91.19,
                                           "seed": 2}).json()
    taus = [v for v in ev["vertices"] if v["name"].startswith("tau")]
    assert len(taus) == 2
    for v in taus:
        assert v["flight_mm"] > 0 and v["betagamma"] > 1


def test_collide_asymmetric_beam(client):
    ev = client.post("/api/collide", json={"beam": "ee", "channel": "z_mumu", "sqrt_s": 91.19,
                                           "asymmetry": 0.5, "seed": 1}).json()
    assert ev["beam"]["beta_z"] == 0.5
    assert ev["beam"]["e_beam1"] > ev["beam"]["e_beam2"]
    assert ev["balance"]["truth_max_dP"] < 1e-12


def test_collide_signal_fraction_zero_gives_background(client):
    ev = client.post("/api/collide", json={"channel": "z_mumu", "signal_fraction": 0.0,
                                           "seed": 4}).json()
    assert ev["truth"]["channel"] == "dy_mumu" and not ev["truth"]["is_signal"]
    assert 60.0 <= ev["truth"]["true_mass"] <= 120.0


@pytest.mark.parametrize("body,needle", [
    ({"beam": "ee", "channel": "h_gammagamma", "sqrt_s": 100.0}, "not reachable"),
    ({"channel": "nonsense"}, "unknown channel"),
    ({"b_field_tesla": 7.0}, "b_field_tesla"),
    ({"sqrt_s": 5000.0}, "sqrt_s"),
    ({"beam": "mumu"}, "beam"),
])
def test_collide_rejects_bad_requests(client, body, needle):
    r = client.post("/api/collide", json=body)
    assert r.status_code == 422
    assert needle in json.dumps(r.json())


# ---- /api/luminosity/step -----------------------------------------------------------


def test_luminosity_run_accumulates_and_fits(client):
    body = {"beam": "pp", "channel": "z_mumu", "sqrt_s": 1000, "signal_fraction": 0.5,
            "n_events": 150, "seed": 2026}
    r1 = client.post("/api/luminosity/step", json=body)
    assert r1.status_code == 200
    s1 = _strict_json(r1)
    assert s1["new_run"] and s1["n_events"] == 150
    h = s1["histogram"]
    assert (h["lo"], h["hi"], h["n_bins"]) == (60.0, 120.0, 60)
    assert len(h["counts"]) == 60 and len(h["edges"]) == 61
    assert sum(h["counts"]) == s1["n_selected"]
    assert s1["event"] is not None and s1["event"]["candidate"] is not None

    s2 = client.post("/api/luminosity/step", json={**body, "run_id": s1["run_id"]}).json()
    assert s2["run_id"] == s1["run_id"] and not s2["new_run"]
    assert s2["n_events"] == 300
    assert all(b >= a for a, b in zip(s1["histogram"]["counts"], s2["histogram"]["counts"]))
    fit = s2["fit"]
    assert fit is not None and fit["success"]
    assert abs(fit["mass"] - s2["pdg_mass"]) < 4 * fit["mass_err"] + 0.3
    assert fit["significance"] > 5 and fit["significance_asimov"] > 5 and fit["discovery"]
    curve = fit["curve"]
    assert len(curve["x"]) == len(curve["total"]) == len(curve["background"])
    assert all(t >= b - 1e-9 for t, b in zip(curve["total"], curve["background"]))
    assert s2["n_signal_true"] + s2["n_background_true"] == 300


def test_luminosity_changed_settings_start_new_run(client):
    body = {"channel": "z_ee", "n_events": 20, "seed": 1}
    s1 = client.post("/api/luminosity/step", json=body).json()
    s2 = client.post("/api/luminosity/step",
                     json={**body, "run_id": s1["run_id"], "b_field_tesla": 2.0}).json()
    assert s2["new_run"] and s2["run_id"] != s1["run_id"] and s2["n_events"] == 20
    # the fit model is NOT part of the run identity: switching refits the same data
    s3 = client.post("/api/luminosity/step", json={**body, "run_id": s2["run_id"],
                                                   "b_field_tesla": 2.0,
                                                   "fit_model": "gaussian"}).json()
    assert s3["run_id"] == s2["run_id"] and s3["n_events"] == 40
    assert client.delete(f"/api/luminosity/{s3['run_id']}").json()["deleted"]


def test_luminosity_rejects_channels_without_peak(client):
    for body in ({"channel": "ww_lnulnu"}, {"channel": "dy_mumu"},
                 {"beam": "ee", "channel": "h_zz_4l", "sqrt_s": 50.0}):
        r = client.post("/api/luminosity/step", json={**body, "n_events": 5})
        assert r.status_code == 422, body


# ---- mystery hooks ------------------------------------------------------------------


def test_mystery_new_hides_truth_and_guess_grades(client):
    r = client.post("/api/mystery/new", json={"channel": "z_mumu", "seed": 9})
    assert r.status_code == 200
    m = _strict_json(r)
    text = r.text
    for leak in ('"name"', '"pdg_id"', '"fate"', '"truth', '"origin"', '"class"', "mu+", "Z0"):
        assert leak not in text, leak
    assert m["clues"]["objects"]["muon"] >= 2
    assert all(t["visible_track"] for t in m["event"]["traces"])

    g = client.post("/api/mystery/guess", json={"challenge_id": m["challenge_id"],
                                                "particles": {"muon": 2}, "channel": "z_mumu"})
    assert g.status_code == 200
    out = g.json()
    assert out["grade"]["score"] == 100 and out["grade"]["truth_channel"] == "z_mumu"
    assert out["reveal"]["truth"]["channel"] == "z_mumu"

    wrong = client.post("/api/mystery/guess", json={"challenge_id": m["challenge_id"],
                                                    "particles": ["electron", "electron"],
                                                    "channel": "z_ee"}).json()
    assert wrong["grade"]["score"] == 25  # right parent (Z), wrong decay & particles


def test_mystery_errors(client):
    assert client.post("/api/mystery/guess", json={"challenge_id": "nope", "particles": [],
                                                   "channel": "z_mumu"}).status_code == 404
    assert client.post("/api/mystery/new", json={"channel": "dy_mumu"}).status_code == 422
    m = client.post("/api/mystery/new", json={"seed": 1}).json()
    bad = client.post("/api/mystery/guess", json={"challenge_id": m["challenge_id"],
                                                  "particles": ["gluino"], "channel": "z_mumu"})
    assert bad.status_code == 422
