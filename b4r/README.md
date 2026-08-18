# SOI ADC biosensor — MEEP/MPB reproduction

Reproduces Alqabandi & Scullion, *J. Opt.* **28** (2026) 045001, using a
400 nm coupling gap (paper's optimum was 700 nm) as requested.

## Requirements
`pymeep` (MEEP + MPB python interface) via conda-forge:
```
conda create -n meep -c conda-forge pymeep pymeep-extras
conda activate meep
```
This could not be installed/run in the current sandbox (no conda-forge
network access here), so none of these scripts have been executed — see the
chat reply for what was and wasn't verified.

## Run order
1. `python fig5_transmissionSpectra.py`
   Sweeps candidate coupling lengths (15–40 um) around the 25 um starting
   guess and picks the one with the deepest through-port dip (= most
   complete power transfer). **Read off the printed "Selected coupling
   length"** — you need it for step 3.
2. `python fig4_modeDispersionMpb.py` — mode-dispersion / anti-crossing plot.
3. Open `fig7fig8_sensitivity.py` and set `COUPLING_LENGTH_UM` to the value
   printed in step 1, then run it — sensitivity fit over RI 1.318–1.400.
4. `python fig6_fieldSnapshots.py` — field snapshots (uses the paper's own
   300 nm/15 um visualisation geometry, not the 400 nm design).

## Files
- `commonConfig.py` — every material/geometry/numerical constant, one place.
- `geometryBuilder.py` — builds the 2D effective-index coupler cross-section.
- `fig4_modeDispersionMpb.py` — Figure 4 (MPB band diagram).
- `fig5_transmissionSpectra.py` — Figure 5 + coupling-length sweep.
- `fig6_fieldSnapshots.py` — Figure 6 (CW field snapshots).
- `fig7fig8_sensitivity.py` — Figures 7 & 8 (sensitivity fit).
