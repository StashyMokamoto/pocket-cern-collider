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
    m = cfg["mystery"]
    assert [c["key"] for c in m["pool"]] == list(server.lab.MYSTERY_POOL) and len(m["pool"]) == 7
    assert {f["key"] for f in m["final_states"]} == {"ee", "mumu", "gammagamma", "4l", "tautau",
                                                       "llnunu"}
    assert m["hint_costs"] == {"1": 5, "2": 15}


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


# ---- Mystery Event Challenge: full workflow ------------------------------------------

TRUTH_LEAKS = ('"name"', '"pdg_id"', '"fate"', '"truth', '"origin"', '"class"', '"hints"',
               '"layer_hints"', '"final_state": "', "mu+", "Z0", "tau", "nu(")


def _new(client, **kw):
    r = client.post("/api/mystery/new", json=kw)
    assert r.status_code == 200, r.text
    return r, _strict_json(r)


@pytest.mark.parametrize("channel", list(server.lab.MYSTERY_POOL))
def test_mystery_blind_event_hides_truth_and_identification(client, channel):
    _, m = _new(client, channel=channel, seed=9, player="blind_test")
    blind = json.dumps({"event": m["event"], "clues": m["clues"]}, ensure_ascii=False)
    for leak in TRUTH_LEAKS:  # (the answer *menu* — pool/final_states — is public, the event is not)
        assert leak not in blind, (channel, leak)
    ev = m["event"]
    assert all(o["kind"] is None for o in ev["objects"])  # the player must identify them
    assert all(t["visible_track"] for t in ev["traces"])  # no neutrino / photon truth lines
    assert ev["vertices"] == []
    assert ev["candidate"] is None or ev["candidate"]["final_state"] is None
    # evidence = raw detector numbers only
    assert set(m["clues"]) == {"layers_lit", "n_tracks", "tracker_hits", "ecal_energy",
                               "hcal_energy", "muon_hits", "met", "met_phi", "large_met",
                               "candidate_mass", "displaced_tracks", "max_d0_mm"}
    assert len(m["pool"]) == 7 and len(m["final_states"]) == 6


def test_mystery_full_workflow_hints_scoring_and_streak(client):
    player = "workflow_player"
    # round 1: perfect claim without hints
    _, m1 = _new(client, channel="z_mumu", seed=21, player=player)
    assert m1["stats"]["rounds"] == 0
    g1 = client.post("/api/mystery/guess", json={"challenge_id": m1["challenge_id"],
                                                 "particles": "mumu", "channel": "z_mumu",
                                                 "player": player}).json()
    assert not g1["already_graded"] and g1["correct"]
    assert g1["final_score"] == g1["grade"]["score"] == 100 and g1["hint_penalty"] == 0
    assert g1["grade"]["truth_category"] == "mumu" and "muon spectrometer" in g1["grade"]["lesson"]
    assert g1["reveal"]["truth"]["channel"] == "z_mumu"  # full MC truth after the claim
    assert g1["stats"]["rounds"] == 1 and g1["stats"]["streak"] == 1

    # round 2: two hint levels cost 5 + 15 points, still a correct (streak-extending) claim
    _, m2 = _new(client, channel="h_gammagamma", seed=22, player=player)
    h1 = client.post("/api/mystery/hint", json={"challenge_id": m2["challenge_id"], "level": 1}).json()
    assert h1["penalty"] == 5 and list(h1["hints"]) == ["1"]
    assert any("ECAL" in x for x in h1["hints"]["1"]) and any("Muon chambers" in x for x in h1["hints"]["1"])
    h2 = client.post("/api/mystery/hint", json={"challenge_id": m2["challenge_id"], "level": 2}).json()
    assert h2["penalty"] == 20 and set(h2["hints"]) == {"1", "2"}
    assert any("photon" in x for x in h2["hints"]["2"])
    g2 = client.post("/api/mystery/guess", json={"challenge_id": m2["challenge_id"],
                                                 "particles": "gammagamma",
                                                 "channel": "h_gammagamma", "player": player}).json()
    assert g2["correct"] and g2["final_score"] == 80 and g2["hint_level"] == 2
    assert g2["stats"]["streak"] == 2 and g2["stats"]["best_streak"] == 2

    # round 3: W⁺W⁻ claimed as Z → μμ — leptons right, neutrinos missed: partial credit
    _, m3 = _new(client, channel="ww_lnulnu", seed=23, player=player)
    g3 = client.post("/api/mystery/guess", json={"challenge_id": m3["challenge_id"],
                                                 "particles": "mumu", "channel": "z_mumu",
                                                 "player": player}).json()
    assert not g3["correct"] and g3["grade"]["particles_partial"] and g3["final_score"] == 25
    assert any("neutrinos" in x for x in g3["grade"]["explanation"])
    assert g3["reveal"]["met"]["truth_met"] > 0  # invisible ν vectors now available to draw
    assert g3["stats"]["streak"] == 0 and g3["stats"]["best_streak"] == 2

    # a second claim on the same event replays the verdict and does not touch the stats
    again = client.post("/api/mystery/guess", json={"challenge_id": m3["challenge_id"],
                                                    "particles": "llnunu",
                                                    "channel": "ww_lnulnu", "player": player}).json()
    assert again["already_graded"] and again["final_score"] == 25
    st = client.get("/api/mystery/stats", params={"player": player}).json()
    assert st["rounds"] == 3 and st["total_score"] == 205 and st["average"] == pytest.approx(68.3)
    assert [h["score"] for h in st["history"]] == [100, 80, 25]
    # hints after the verdict are free and complete
    late = client.post("/api/mystery/hint", json={"challenge_id": m3["challenge_id"], "level": 2}).json()
    assert set(late["hints"]) == {"1", "2"} and late["penalty"] == 0


def test_mystery_tau_event_explains_displacement(client):
    _, m = _new(client, channel="z_tautau", seed=3, player="tau_player")
    g = client.post("/api/mystery/guess", json={"challenge_id": m["challenge_id"],
                                                "particles": "tautau", "channel": "z_tautau",
                                                "player": "tau_player"}).json()
    assert g["correct"] and g["final_score"] == 100
    assert "displaced vertex" in g["grade"]["lesson"]
    assert any("τ flight lengths" in x for x in g["grade"]["explanation"])
    taus = [v for v in g["reveal"]["vertices"] if v["name"].startswith("tau")]
    assert len(taus) == 2  # revealed only after the claim
    assert len([t for t in g["reveal"]["traces"] if t["class"] == "neutrino"]) >= 2


def test_mystery_errors(client):
    assert client.post("/api/mystery/guess", json={"challenge_id": "nope", "particles": "ee",
                                                   "channel": "z_mumu"}).status_code == 404
    assert client.post("/api/mystery/hint", json={"challenge_id": "nope"}).status_code == 404
    assert client.post("/api/mystery/new", json={"channel": "dy_mumu"}).status_code == 422
    assert client.post("/api/mystery/new", json={"player": "bad id!"}).status_code == 422
    assert client.get("/api/mystery/stats", params={"player": "never_played"}).status_code == 404
    _, m = _new(client, seed=1)
    cid = m["challenge_id"]
    assert client.post("/api/mystery/hint", json={"challenge_id": cid, "level": 3}).status_code == 422
    for bad in ({"particles": "gluinos", "channel": "z_mumu"},
                {"particles": ["gluino"], "channel": "z_mumu"},
                {"particles": "ee", "channel": "dy_ee"}):
        assert client.post("/api/mystery/guess", json={"challenge_id": cid, **bad}).status_code == 422
    # explicit particle counts are still accepted
    ok = client.post("/api/mystery/guess", json={"challenge_id": cid, "particles": {"muon": 2},
                                                 "channel": "z_mumu"})
    assert ok.status_code == 200


def test_luminosity_lorentzian_peak_fit(client):
    r = client.post("/api/luminosity/step", json={"channel": "z_ee", "n_events": 400, "seed": 7,
                                                  "signal_fraction": 0.6, "fit_model": "lorentzian"})
    fit = r.json()["fit"]
    assert fit["model"] == "lorentzian" and fit["success"]
    assert abs(fit["mass"] - pe.pdg(pe.Z0).mass) < 1.0 and fit["width"] > 0


# ---- UI ------------------------------------------------------------------------------


def test_ui_contains_mystery_and_snapshot_controls(client):
    html = client.get("/").text
    for needle in ('id="modeSeg"', 'id="mNewBtn"', 'id="pillsA"', 'id="pillsB"', 'id="claimBtn"',
                   'id="evList"', 'id="sigTable"', 'id="hint1"', 'id="hint2"', 'id="resultBox"',
                   'id="snapBtn"', "/api/mystery/new", "/api/mystery/hint", "/api/mystery/guess",
                   "toBlob", "SUBMIT DISCOVERY CLAIM", "NEW MYSTERY COLLISION"):
        assert needle in html, needle
    assert "Milestone 4" not in html  # the locked placeholder is gone


def _find_chrome():
    import shutil
    from pathlib import Path
    for c in (shutil.which("chrome"), shutil.which("google-chrome"), shutil.which("chromium"),
              shutil.which("msedge"),
              r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"):
        if c and Path(c).exists():
            return c
    return None


@pytest.mark.skipif(_find_chrome() is None, reason="no headless Chrome/Edge available")
def test_headless_browser_plays_a_full_mystery_round(tmp_path):
    """End-to-end: real server + real browser — load a blind Z→ττ event, take a hint, pick
    both answers, submit the claim; the page must reach the 'revealed' state error-free."""
    import re
    import socket
    import subprocess
    import threading
    import time
    import urllib.request

    import uvicorn

    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        url = (f"http://127.0.0.1:{port}/?mode=mystery&mchannel=z_tautau&mseed=3&hints=1"
               "&mfs=tautau&mguess=z_tautau&autosubmit=1&anim=0&view=split")
        out = subprocess.run([_find_chrome(), "--headless=new", "--disable-gpu",
                              f"--user-data-dir={tmp_path}", "--window-size=1600,950",
                              "--virtual-time-budget=15000", "--dump-dom", url],
                             capture_output=True, text=True, encoding="utf-8", timeout=120)
        dom = out.stdout
        body = re.search(r"<body[^>]*>", dom).group(0)
        assert 'data-state="mystery-revealed"' in body, body
        assert "data-errors" not in body, body
        assert "DISCOVERY CONFIRMED" in dom and "Physics hint" in dom
        assert re.search(r'class="pill on truth"[^>]*data-v="tautau"', dom)
    finally:
        srv.should_exit = True
        th.join(timeout=10)
