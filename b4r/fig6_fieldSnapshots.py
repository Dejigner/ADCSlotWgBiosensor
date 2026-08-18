"""
fig6_fieldSnapshots.py
========================
Reproduces figure 6: Ey field-distribution snapshots from a continuous-wave
source, using the SAME non-optimised visualisation geometry the paper uses
for this figure only (300 nm gap, 15 um coupling length - stated explicitly
in the figure 6 caption) so the coupling is easy to see by eye:
  (a) source at the coupling wavelength (1550 nm)  -> light crosses over
  (b) source at a shorter wavelength (1200 nm)      -> light stays in the
      straight (bus) waveguide

Run:
    python fig6_fieldSnapshots.py
"""

import matplotlib.pyplot as plt
import meep as mp

import commonConfig as cfg
from geometryBuilder import buildCellAndGeometry


VISUALISATION_GAP = 0.300        # um, matches figure 6 caption (not 400 nm)
VISUALISATION_LENGTH = 15.0      # um, matches figure 6 caption


def snapshotAtWavelength(wavelengthUm, outputFile):
    cellSize, geometry, strY, leftRailY, rightRailY = buildCellAndGeometry(
        VISUALISATION_LENGTH, VISUALISATION_GAP, includeSlotArm=True)

    sourceX = -cellSize.x / 2 + cfg.pmlThickness + 0.3
    sources = [mp.Source(mp.ContinuousSource(frequency=1 / wavelengthUm),
                          component=mp.Ey,
                          center=mp.Vector3(sourceX, strY, 0),
                          size=mp.Vector3(0, 1.2 * cfg.straightWaveguideWidth, 0))]

    sim = mp.Simulation(cell_size=cellSize,
                         resolution=cfg.fdtdResolution,
                         boundary_layers=[mp.PML(cfg.pmlThickness)],
                         geometry=geometry,
                         sources=sources,
                         default_material=cfg.claddingMedium(cfg.waterIndex))

    # Run long enough for the continuous wave to reach steady state across
    # the whole cell: propagation time ~ effective_index * length / c.
    runTime = 3 * cellSize.x * cfg.siliconEffIndex
    sim.run(until=runTime)

    sim.plot2D(fields=mp.Ey)
    plt.title(f"Ey field, source wavelength = {wavelengthUm*1000:.0f} nm")
    plt.tight_layout()
    plt.savefig(outputFile, dpi=200)
    plt.close()
    sim.reset_meep()
    print(f"Saved {outputFile}")


if __name__ == "__main__":
    snapshotAtWavelength(1.550, "fig6a_coupling_1550nm.png")
    snapshotAtWavelength(1.200, "fig6b_noCoupling_1200nm.png")
