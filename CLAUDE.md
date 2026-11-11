# Pocket CERN Collider — Architecture & Milestone Contract

A local, educational particle-collider simulator: exact relativistic event generation,
an ATLAS/CMS-style detector simulation, an invariant-mass analysis lab, and a dark-mode
"CERN Control Room" web UI with a Mystery Particle discovery game.

## Conventions (apply everywhere)

- **Natural units:** energies, momenta and masses in **GeV**, with c = ħ = 1.
  Lengths (vertices, cτ, detector radii) in **mm**. Magnetic field in **T**.
- 4-momentum ordering is always **(E, px, py, pz)**; metric signature (+, −, −, −).
- Beam axis is **ẑ**. Beam 1 (e⁻ or p) travels along +z, beam 2 (e⁺ or p) along −z.
- PDG data comes **only** from the Scikit-HEP `particle` library (live PDG tables,
  converted MeV → GeV). Never hard-code masses/widths that the library provides.
- Python 3.12, virtual environment in `.venv/` managed with `uv`
  (`uv venv --python 3.12`; deps listed in `requirements.txt`).
- Tests live in `tests/` and run with `.venv/Scripts/python -m pytest` (Windows) /
  `.venv/bin/python -m pytest`.

## Strict 4-File Modular Architecture

Source code is confined to these four modules (plus one static HTML page). Do not add
new source modules; tests in `tests/` and config files are the only extras.

1. **`physics_engine.py`** — PDG database integration (`particle` library), relativistic
   4-momentum vectors (E, px, py, pz) in natural units (GeV, c = 1), relativistic
   Breit–Wigner resonance sampling, exact 2-body (and 3-body phase-space) decays with
   cascading Lorentz boosts Λ(β), displaced decay vertices, beam configuration
   (e⁺e⁻ / pp, √s from 2 to 1000 GeV, optional asymmetric boost along ẑ), golden
   discovery channels, and Drell–Yan / QED / diboson continuum background generation.
   Self-test: `python physics_engine.py`.
2. **`detector_sim.py`** — Concentric ATLAS/CMS-style transverse & 3D cylindrical
   geometry (beam pipe, silicon vertex detector/tracker in an axial B_z = 3.8 T field,
   ECAL, HCAL, muon spectrometer), Larmor helical trajectory points, displaced secondary
   vertices (cτ·βγ), realistic energy/momentum resolution smearing, and Missing
   Transverse Energy (MET / E̸_T) from escaping neutrinos.
3. **`invariance_lab.py`** — Lead-Scientist analysis engine: reconstructs candidate
   invariant mass M_inv = √((ΣE)² − |Σp⃗|²) from smeared detector objects, manages live
   histogram binning + Lorentzian / Breit–Wigner curve fitting over background, and
   powers the "Mystery Event Challenge" mode.
4. **`app.py`** (plus a single **`static/index.html`**) — FastAPI local server +
   high-FPS dark-mode interactive CERN control-room UI (ATLAS transverse cross-section
   canvas, beam/energy controls, Luminosity Run mode, live invariant-mass histogram,
   and the Mystery Particle educational game).

Dependency direction is strictly one-way:
`physics_engine` ← `detector_sim` ← `invariance_lab` ← `app`.

## physics_engine.py public API (stable contract for later milestones)

- `pdg(pdg_id) -> ParticleSpec` — cached PDG lookup (mass, width, charge, cτ, spin, kind).
- `FourVector` — frozen dataclass with `mass`, `pt`, `eta`, `phi`, `rapidity`, `boost(...)`,
  `FourVector.sum(...)` (compensated `math.fsum`).
- `lorentz_boost_matrix(beta_vec, gamma=None)` — exact 4×4 Λ(β⃗).
- `sample_breit_wigner_mass(...)`, `relativistic_breit_wigner_pdf(...)`.
- `BeamConfig(kind, sqrt_s, beta_z)` / `BeamConfig.from_beam_energies(kind, e1, e2)`.
- `CollisionGenerator(beam, seed).generate(channel, **options) -> Event`.
- `CHANNELS` registry (signal channels know their background channel + mass window).
- `Event` — truth record of `GenParticle`s (status, mother/daughters, production &
  decay vertices), `final_state()`, `visible_final_state()`, `neutrinos()`,
  `truth_missing_pt()`, `conservation_residual()`, `table()`.

## Milestone Checklist

- [x] **Milestone 1 (25%):** Setup + `CLAUDE.md` + `physics_engine.py` built, verified with conservation tests, and committed.
- [ ] **Milestone 2 (50%):** `detector_sim.py` + `invariance_lab.py` built, verified end-to-end in terminal, and committed.
- [ ] **Milestone 3 (75%):** `app.py` + `static/index.html` interactive CERN Control Room & ATLAS Event Display live on localhost and committed.
- [ ] **Milestone 4 (100%):** "Lead Scientist / Mystery Event" educational game mode + Lorentzian resonance peak fitter + polished GitHub `README.md` completed and committed.

Work proceeds strictly one milestone at a time; stop for review after each milestone.
