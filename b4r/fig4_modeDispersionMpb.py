"""
fig4_modeDispersionMpb.py
===========================
Reproduces figure 4: mode-dispersion (frequency vs wave vector) curves of
the isolated straight and slot waveguides, computed with MPB, plus the
cladding light line. The wavelength where the two fundamental-mode curves
cross is the phase-matching / coupling wavelength used to design the ADC.

Unlike the MEEP transmission-spectrum simulation (2D, effective index),
this MPB dispersion analysis uses the TRUE cross-section (real 220 nm Si
height, bulk Si index 3.4757, SiO2 substrate, water cladding) because that
is what the paper describes for the frequency-domain analysis in Section 3.

Run:
    python fig4_modeDispersionMpb.py
"""

import numpy as np
import matplotlib.pyplot as plt
import meep as mp
import meep.mpb as mpb

import commonConfig as cfg

# ---------------------------------------------------------------------------
# Section 1: cross-section builder (real 3D-style waveguide cross section)
# ---------------------------------------------------------------------------
def buildCrossSection(coreWidths, gapInCore=None):
    """
    coreWidths : single width (straight arm) or (railWidth, railWidth) for
                 the slot arm, each centered so the whole feature sits at y=0.
    gapInCore  : slot gap width, or None for a solid (straight) core.
    """
    substrateThickness = 2.0  # um, thick enough to behave as a half-space
    claddingHeight = 2.0      # um, padding above the slab
    sizeY = 3.0                # um, transverse (in-plane) simulation width
    sizeZ = substrateThickness + cfg.siliconThickness + claddingHeight

    geometry = [
        # SiO2 substrate half-space (z < 0)
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, substrateThickness),
                 center=mp.Vector3(0, 0, -substrateThickness / 2),
                 material=cfg.silicaMedium),
    ]

    zCenter = cfg.siliconThickness / 2  # Si slab occupies z in [0, thickness]

    if gapInCore is None:
        geometry.append(
            mp.Block(size=mp.Vector3(mp.inf, coreWidths, cfg.siliconThickness),
                      center=mp.Vector3(0, 0, zCenter),
                      material=mp.Medium(index=cfg.siliconBulkIndex)))
    else:
        railWidth = coreWidths
        offset = railWidth / 2 + gapInCore / 2
        for sign in (+1, -1):
            geometry.append(
                mp.Block(size=mp.Vector3(mp.inf, railWidth, cfg.siliconThickness),
                          center=mp.Vector3(0, sign * offset, zCenter),
                          material=mp.Medium(index=cfg.siliconBulkIndex)))

    lattice = mp.Lattice(size=mp.Vector3(0, sizeY, sizeZ))
    return lattice, geometry


# ---------------------------------------------------------------------------
# Section 2: run MPB over a range of wave vectors for one waveguide
# ---------------------------------------------------------------------------
def computeDispersion(lattice, geometry, kxList, numBands=2, resolution=64,
                       claddingIndex=cfg.waterIndex):
    # resolution=64 px/um (rather than MEEP's 30) so the 50 nm slot gap is
    # resolved by ~3 pixels - MPB is an eigensolver and comparatively cheap,
    # so this extra resolution is affordable and needed for the slot mode.
    modeSolver = mpb.ModeSolver(
        geometry_lattice=lattice,
        geometry=geometry,
        k_points=[mp.Vector3(kx, 0, 0) for kx in kxList],
        resolution=resolution,
        num_bands=numBands,
        default_material=cfg.claddingMedium(claddingIndex))
    modeSolver.run()
    freqsByBand = np.array(modeSolver.all_freqs)  # shape (len(kxList), numBands)
    return freqsByBand


# ---------------------------------------------------------------------------
# Section 3: main - straight vs slot waveguide dispersion + light line
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    kxList = np.linspace(0.8, 1.7, 40)  # normalised wave vector, 1/um

    straightLattice, straightGeometry = buildCrossSection(cfg.straightWaveguideWidth)
    slotLattice, slotGeometry = buildCrossSection(cfg.slotRailWidth, cfg.slotGapWidth)

    print("Computing straight-waveguide dispersion...")
    straightFreqs = computeDispersion(straightLattice, straightGeometry, kxList, numBands=2)

    print("Computing slot-waveguide dispersion...")
    slotFreqs = computeDispersion(slotLattice, slotGeometry, kxList, numBands=1)

    lightLine = kxList / cfg.waterIndex

    # Locate the phase-matching (anti-crossing) point: where the two
    # fundamental-mode curves are closest together.
    freqDifference = np.abs(straightFreqs[:, 0] - slotFreqs[:, 0])
    crossingIndex = np.argmin(freqDifference)
    crossingKx = kxList[crossingIndex]
    crossingFreq = straightFreqs[crossingIndex, 0]
    crossingWavelengthNm = 1000 / crossingFreq
    print(f"Closest approach at kx = {crossingKx:.3f}, "
          f"normalised frequency = {crossingFreq:.4f}, "
          f"wavelength = {crossingWavelengthNm:.1f} nm")

    plt.figure(figsize=(7, 5))
    plt.plot(kxList, straightFreqs[:, 0], 'g-', label="Through port (straight, TE0)")
    plt.plot(kxList, slotFreqs[:, 0], 'r-', label="Cross port (slot, TE0)")
    plt.plot(kxList, straightFreqs[:, 1], 'b-', label="Higher mode (straight, TE1)")
    plt.plot(kxList, lightLine, 'k--', label="Light line (water cladding)")
    plt.axhline(1 / cfg.centerWavelength, color='gray', linestyle=':',
                label="Frequency corresponding to 1550 nm")
    plt.plot(crossingKx, crossingFreq, 'x', color='cyan', markersize=10,
              label=f"Anti-crossing ({crossingWavelengthNm:.0f} nm)")
    plt.xlabel("Normalised wave vector (1/um)")
    plt.ylabel("Normalised frequency (1/um)")
    plt.title("Mode dispersion: straight vs slot waveguide")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("fig4_modeDispersion.png", dpi=200)
    print("Saved fig4_modeDispersion.png")
