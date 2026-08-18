"""
commonConfig.py
================
Shared physical / geometric / numerical parameters for the SOI asymmetric
directional-coupler (ADC) biosensor, reproduced from:

  Alqabandi & Scullion, "SOI asymmetrical directional coupler based photonic
  biosensor for 1550 nm optical range", J. Opt. 28 (2026) 045001.

All the values below are taken directly from the paper's Section 3
("Simulation method") unless a comment says otherwise. Two values were
changed on purpose at the user's request (gap width and coupling length) -
see the flagged lines below.
"""

import meep as mp

# ---------------------------------------------------------------------------
# Material refractive indices (Section 3)
# ---------------------------------------------------------------------------
siliconBulkIndex   = 3.4757   # bulk Si index used only for the 1D z-solver
siliconEffIndex    = 2.842    # effective index of 220 nm Si slab (2D FDTD)
silicaIndex        = 1.444    # SiO2 substrate / cladding (spectrometer case)
waterIndex         = 1.318    # reference sensing cladding at RI = 1.318

siliconMedium = mp.Medium(index=siliconEffIndex)
silicaMedium  = mp.Medium(index=silicaIndex)

def claddingMedium(refractiveIndex=waterIndex):
    """Analyte / cladding medium (water by default, swept for sensitivity)."""
    return mp.Medium(index=refractiveIndex)

# ---------------------------------------------------------------------------
# Waveguide cross-section geometry (Section 3 / figure 1)
# ---------------------------------------------------------------------------
siliconThickness   = 0.220    # um, physical Si slab thickness (used only
                               # for the 1D mode solver, not in the 2D FDTD)

straightWaveguideWidth = 0.350  # um, straight (bus) arm
slotOuterWidth          = 0.550  # um, outer width of the slotted arm
slotGapWidth             = 0.050  # um, nano-slot width
slotRailWidth            = (slotOuterWidth - slotGapWidth) / 2  # each rail

# NOTE (deviation from paper, requested by user):
#   The paper's optimised device uses a 700 nm edge-to-edge gap and a
#   200 um coupling length (see Table 1 / figure 7 caption). Here we use a
#   400 nm gap as instructed, which requires a MUCH shorter coupling length
#   because evanescent coupling strength grows quickly as the gap shrinks.
gapWidth              = 0.400   # um, edge-to-edge separation (paper: 0.700)
couplingLengthGuess   = 25.0    # um, starting guess (paper: 200 for 700 nm gap)

# ---------------------------------------------------------------------------
# Simulation numerics (Section 3)
# ---------------------------------------------------------------------------
fdtdResolution   = 30        # pixels/um, matches paper's convergence choice
pmlThickness     = 1.0       # um
leadLength       = 3.0       # um, straight input/output leads either side
                              # of the coupling region (not stated explicitly
                              # in the paper - needed for source/eigenmode
                              # launch and PML separation; see notes)

wavelengthMin    = 1.0       # um, transmission-spectrum sweep (figure 5)
wavelengthMax    = 2.0       # um
centerWavelength = 1.55      # um, Gaussian source center
numFrequencies   = 400       # spectral resolution of the flux monitors

# Refractive-index sweep used for the sensitivity study (figures 7 & 8)
sensitivityRiList = [1.318, 1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.38, 1.39, 1.40]
