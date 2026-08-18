"""
fig5_transmissionSpectra.py
============================
Reproduces figure 5 of the paper: normalised through-port and cross-port
power vs wavelength (1-2 um) for the optimised coupler geometry.

Also implements the length-optimisation sweep the user asked for: since the
paper only reports a coupling length for the 700 nm gap design (200 um), a
short sweep around the ~25 um starting guess is run here to find the length
that gives maximum power transfer (deepest through-port dip) for the 400 nm
gap requested by the user. See the notes in the chat reply for why this
sweep - not a single hard-coded number - is the correct way to pin this
down.

Run:
    python fig5_transmissionSpectra.py
"""

import numpy as np
import matplotlib.pyplot as plt
import meep as mp

import commonConfig as cfg
from geometryBuilder import buildCellAndGeometry, transverseLayout


# ---------------------------------------------------------------------------
# Section 1: core simulation routine (one FDTD run -> flux vs frequency)
# ---------------------------------------------------------------------------
def runFluxSimulation(deviceLength, gapWidth, claddingIndex, includeSlotArm):
    """
    Runs a single broadband FDTD simulation and returns the frequency array
    together with the through-port and cross-port flux spectra (raw, not yet
    normalised). If includeSlotArm is False, only the through-port flux is
    meaningful (used for the incident-power reference run).
    """
    cellSize, geometry, strY, leftRailY, rightRailY = buildCellAndGeometry(
        deviceLength, gapWidth, includeSlotArm)

    freqMin = 1 / cfg.wavelengthMax
    freqMax = 1 / cfg.wavelengthMin
    freqCenter = 1 / cfg.centerWavelength
    freqWidth = freqMax - freqMin

    sourceX = -cellSize.x / 2 + cfg.pmlThickness + 0.3
    monitorX = cellSize.x / 2 - cfg.pmlThickness - 0.3

    sources = [mp.Source(mp.GaussianSource(frequency=freqCenter, fwidth=freqWidth),
                          component=mp.Ey,
                          center=mp.Vector3(sourceX, strY, 0),
                          size=mp.Vector3(0, 1.2 * cfg.straightWaveguideWidth, 0))]

    sim = mp.Simulation(cell_size=cellSize,
                         resolution=cfg.fdtdResolution,
                         boundary_layers=[mp.PML(cfg.pmlThickness)],
                         geometry=geometry,
                         sources=sources,
                         default_material=cfg.claddingMedium(claddingIndex))

    throughRegion = mp.FluxRegion(center=mp.Vector3(monitorX, strY, 0),
                                   size=mp.Vector3(0, 1.5 * cfg.straightWaveguideWidth, 0))
    throughMonitor = sim.add_flux(freqCenter, freqWidth, cfg.numFrequencies, throughRegion)

    crossMonitor = None
    if includeSlotArm:
        crossCenterY = (leftRailY + rightRailY) / 2
        crossRegion = mp.FluxRegion(center=mp.Vector3(monitorX, crossCenterY, 0),
                                     size=mp.Vector3(0, 1.5 * cfg.slotOuterWidth, 0))
        crossMonitor = sim.add_flux(freqCenter, freqWidth, cfg.numFrequencies, crossRegion)

    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ey, mp.Vector3(monitorX, strY, 0), 1e-4))

    freqs = np.array(mp.get_flux_freqs(throughMonitor))
    throughFlux = np.array(mp.get_fluxes(throughMonitor))
    crossFlux = np.array(mp.get_fluxes(crossMonitor)) if crossMonitor else None

    sim.reset_meep()
    return freqs, throughFlux, crossFlux


# ---------------------------------------------------------------------------
# Section 2: normalised spectra for a given device length / gap / cladding
# ---------------------------------------------------------------------------
def normalisedSpectra(deviceLength, gapWidth=cfg.gapWidth, claddingIndex=cfg.waterIndex):
    """Returns wavelengths (nm), throughRatio, crossRatio for one coupler."""
    freqsRef, incidentFlux, _ = runFluxSimulation(
        deviceLength, gapWidth, claddingIndex, includeSlotArm=False)
    freqs, throughFlux, crossFlux = runFluxSimulation(
        deviceLength, gapWidth, claddingIndex, includeSlotArm=True)

    wavelengthsNm = 1000.0 / freqs  # meep freq is in 1/um -> wavelength in um -> nm
    throughRatio = throughFlux / incidentFlux
    crossRatio = crossFlux / incidentFlux
    return wavelengthsNm, throughRatio, crossRatio


# ---------------------------------------------------------------------------
# Section 3: coupling-length optimisation sweep
# ---------------------------------------------------------------------------
def optimiseCouplingLength(candidateLengths, gapWidth=cfg.gapWidth,
                            claddingIndex=cfg.waterIndex):
    """
    Runs the coupler at each candidate length and reports the depth of the
    through-port dip (i.e. how close to full power transfer is achieved).
    The best length is the one with the lowest minimum through-port ratio.
    """
    results = []
    for length in candidateLengths:
        _, throughRatio, _ = normalisedSpectra(length, gapWidth, claddingIndex)
        dipDepth = np.min(throughRatio)
        results.append((length, dipDepth))
        print(f"  length = {length:5.1f} um  ->  min through-port power = {dipDepth:.4f}")

    bestLength = min(results, key=lambda pair: pair[1])[0]
    return bestLength, results


# ---------------------------------------------------------------------------
# Section 4: main - run the optimisation, then plot figure 5
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Sweeping candidate coupling lengths around the 25 um starting guess...")
    candidateLengths = [15, 20, 25, 30, 35, 40]
    bestLength, sweepResults = optimiseCouplingLength(candidateLengths)
    print(f"\nSelected coupling length: {bestLength} um (deepest through-port dip)")

    wavelengthsNm, throughRatio, crossRatio = normalisedSpectra(bestLength)

    plt.figure(figsize=(7, 5))
    plt.plot(wavelengthsNm / 1000, throughRatio, label="Through Port")
    plt.plot(wavelengthsNm / 1000, crossRatio, label="Cross Port")
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Power ratio")
    plt.title(f"Transmission Spectra (gap = {cfg.gapWidth*1000:.0f} nm, "
              f"length = {bestLength:.0f} um)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig5_transmissionSpectra.png", dpi=200)
    print("Saved fig5_transmissionSpectra.png")
