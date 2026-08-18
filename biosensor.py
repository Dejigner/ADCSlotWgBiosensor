"""
================================================================================
SOI ASYMMETRIC DIRECTIONAL COUPLER (ADC) BIOSENSOR - MEEP / MPB REPRODUCTION
--------------------------------------------------------------------------------
Source paper : Alqabandi & Scullion, "SOI asymmetrical directional coupler
                based photonic biosensor for 1550 nm optical range",
                J. Opt. 28, 045001 (2026)

This script reproduces every figure in the paper:
    Fig. 1  - device cross-section schematic
    Fig. 2  - MEEP simulation-domain schematic
    Fig. 4  - mode dispersion (MPB, straight vs slot waveguide)
    Fig. 5  - broadband transmission spectra (MEEP FDTD)
    Fig. 6  - CW electric-field snapshots (coupled / non-coupled)
    Fig. 7  - dip-wavelength shift for two cladding samples
    Fig. 8  - sensitivity plot (coupled wavelength vs refractive index)
    Table 1 - sensitivity / FWHM / FOM vs coupling-gap width
    Fig. 9  - coupled wavelength vs slot width (on-chip spectrometer)
    Fig. 10 - coupled wavelength vs temperature (thermal tuning)

Requires: meep, meep.mpb (MIT Photonic-Bands python bindings), numpy, matplotlib
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import meep as mp
from meep import mpb

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


# ==============================================================================
# SECTION 1 : MATERIAL AND GEOMETRY PARAMETERS  (paper, Section 3)
# ==============================================================================
nSilicon = 3.4757          # Si index at 1550 nm
nOxide = 1.444              # SiO2 substrate index
nWaterClad = 1.318          # water cladding, reference sensing index (RIU)
waveguideThk = 0.220         # um, SOI layer thickness

straightWidth = 0.350        # um, straight (strip) arm width
slotArmWidth = 0.550         # um, slot arm outer width
slotGapWidth = 0.050         # um, nanoslot gap
couplerGap = 0.700           # um, edge-to-edge separation between the two arms
designWvl = 1.55             # um, telecom design wavelength

effIndexSi = 2.842           # 1D-mode-solver effective index (paper value) used
                              # to fold the vertical (z) confinement into the
                              # 2D FDTD model, replacing the bulk Si index

siliconMed2D = mp.Medium(index=effIndexSi)   # used only inside 2D FDTD (MEEP)
siliconMed3D = mp.Medium(index=nSilicon)     # real Si index, used in MPB
oxideMed = mp.Medium(index=nOxide)
waterMed = mp.Medium(index=nWaterClad)

resolutionFDTD = 30          # pixels/um (paper: 25 adequate, 30 used for robustness)
pmlThickness = 1.0            # um  -- not stated in paper, standard MEEP default assumed


# ==============================================================================
# SECTION 2 : FIGURE 1 - DEVICE CROSS-SECTION SCHEMATIC
# ==============================================================================
def plotCrossSection():
    """Recreates Fig. 1: straight + slot waveguide cross-section with TE field cartoon."""
    fig, ax = plt.subplots(figsize=(6, 3))

    strX0 = -0.9
    ax.add_patch(plt.Rectangle((strX0, 0), straightWidth, waveguideThk, color="firebrick"))

    slotX0 = 0.5
    armW = (slotArmWidth - slotGapWidth) / 2
    ax.add_patch(plt.Rectangle((slotX0, 0), armW, waveguideThk, color="firebrick"))
    ax.add_patch(plt.Rectangle((slotX0 + armW + slotGapWidth, 0), armW, waveguideThk,
                                color="firebrick"))

    ax.add_patch(plt.Rectangle((-1.5, -0.3), 3.5, 0.3, color="lightsteelblue"))  # SiO2
    ax.set_xlim(-1.5, 2.0)
    ax.set_ylim(-0.4, 0.6)
    ax.set_title("Fig. 1 - ADC cross-section (straight arm left, slot arm right)")
    ax.text(strX0, 0.35, "Straight\nwaveguide", ha="left", fontsize=8)
    ax.text(slotX0, 0.35, "Slot\nwaveguide", ha="left", fontsize=8)
    ax.text(-1.4, -0.15, "SiO2", fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.savefig(f"{OUT_DIR}/fig1_cross_section.png", dpi=200)
    plt.close(fig)


# ==============================================================================
# SECTION 3 : FIGURE 2 - MEEP SIMULATION DOMAIN SCHEMATIC
# ==============================================================================
def plotSimulationSchematic(length=200.0):
    """Recreates Fig. 2: source, waveguides, PML and flux-monitor layout."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.add_patch(plt.Rectangle((-length / 2, -1.0), length, 2.0,
                                facecolor="none", edgecolor="green", hatch="//"))
    ax.add_patch(plt.Rectangle((-length / 2, -0.15), length, 0.1, color="black"))
    ax.axvline(-length / 2 + 2, color="red", label="Source")
    ax.axvline(length / 2 - 2, color="blue", label="Flux monitor")
    ax.annotate("Input light", xy=(-length / 2 + 2, 0.3), xytext=(-length / 2 + 8, 0.4),
                arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(-length / 2 - 5, length / 2 + 5)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title("Fig. 2 - MEEP simulation domain (source, waveguide, PML, flux monitor)")
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")

    fig.savefig(f"{OUT_DIR}/fig2_simulation_schematic.png", dpi=200)
    plt.close(fig)


# ==============================================================================
# SECTION 4 : FIGURE 4 - MODE DISPERSION (MPB FREQUENCY-DOMAIN ANALYSIS)
# ==============================================================================
def solveWaveguideDispersion(waveguideWidth, slotted, kMin, kMax, numK,
                              numBands, claddingMed=waterMed):
    """
    Cross-sectional MPB solve for one waveguide (strip or slot), sweeping the
    normalised wave vector to build the dispersion curve used in paper Fig. 4.
    """
    cellX, cellY = 3.0, 3.0     # um, supercell size - not given in paper, assumed
    lattice = mp.Lattice(size=mp.Vector3(cellX, cellY))

    geometry = []
    if not slotted:
        geometry.append(mp.Block(size=mp.Vector3(waveguideWidth, waveguideThk, mp.inf),
                                  center=mp.Vector3(0, 0), material=siliconMed3D))
    else:
        armWidth = (waveguideWidth - slotGapWidth) / 2
        offset = (armWidth + slotGapWidth) / 2
        for sign in (+1, -1):
            geometry.append(mp.Block(size=mp.Vector3(armWidth, waveguideThk, mp.inf),
                                      center=mp.Vector3(sign * offset, 0),
                                      material=siliconMed3D))

    kPoints = mp.interpolate(numK, [mp.Vector3(kMin), mp.Vector3(kMax)])

    modeSolver = mpb.ModeSolver(geometry_lattice=lattice,
                                 geometry=geometry,
                                 k_points=kPoints,
                                 resolution=32,          # px/um, not stated, assumed
                                 num_bands=numBands,
                                 default_material=claddingMed)
    modeSolver.run_te()   # TE polarisation, per paper

    freqs = np.array(modeSolver.all_freqs)
    kMags = np.array([k.norm() for k in kPoints])
    return kMags, freqs


def plotModeDispersion():
    """Recreates Fig. 4(a)/(b): dispersion + zoom around the anti-crossing point."""
    kMin, kMax, numK = 0.8, 1.7, 40   # range chosen to match paper's Fig. 4(a) axis
    kStr, fStr = solveWaveguideDispersion(straightWidth, slotted=False,
                                           kMin=kMin, kMax=kMax, numK=numK, numBands=1)
    kSlot, fSlot = solveWaveguideDispersion(slotArmWidth, slotted=True,
                                             kMin=kMin, kMax=kMax, numK=numK, numBands=2)
    lightLine = kStr / nWaterClad   # lower-cladding light line

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(kStr, fStr[:, 0], "g-", label="Through port (straight)")
    axs[0].plot(kSlot, fSlot[:, 0], "r-", label="Cross port (slot)")
    axs[0].plot(kSlot, fSlot[:, 1], color="navy", label="Higher mode")
    axs[0].plot(kStr, lightLine, "k--", label="Light line")
    axs[0].axhline(1 / designWvl, color="steelblue", ls=":", label="1550 nm")
    axs[0].set_xlabel("Normalised wave vector (a.u.)")
    axs[0].set_ylabel("Normalised frequency (a.u.)")
    axs[0].legend(fontsize=7)
    axs[0].set_title("(a) Full dispersion")

    axs[1].plot(kStr, fStr[:, 0], "g-")
    axs[1].plot(kSlot, fSlot[:, 0], "r-")
    axs[1].plot(kSlot, fSlot[:, 1], color="navy")
    axs[1].plot(kStr, lightLine, "k--")
    axs[1].set_xlim(0.95, 1.15)
    axs[1].set_title("(b) Zoom near anti-crossing")
    axs[1].set_xlabel("Normalised wave vector (a.u.)")

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig4_mode_dispersion.png", dpi=200)
    plt.close(fig)


# ==============================================================================
# SECTION 5 : FDTD COUPLER ENGINE (used by Fig. 5, 6, 7, 8, Table 1, Fig. 9, 10)
# ==============================================================================
def buildCouplerGeometry(slotWidth, gap, length):
    """Two waveguides (straight + slot) running along x, separated by `gap`."""
    armWidth = (slotArmWidth - slotWidth) / 2
    yStraight = -(gap + straightWidth) / 2
    ySlot = (gap + slotArmWidth) / 2

    geometry = [mp.Block(size=mp.Vector3(length, straightWidth, mp.inf),
                          center=mp.Vector3(0, yStraight), material=siliconMed2D)]
    for sign in (+1, -1):
        geometry.append(mp.Block(
            size=mp.Vector3(length, armWidth, mp.inf),
            center=mp.Vector3(0, ySlot + sign * (armWidth + slotWidth) / 2),
            material=siliconMed2D))
    return geometry, yStraight, ySlot


def runCouplerFDTD(claddingIndex=nWaterClad, slotWidth=slotGapWidth, gap=couplerGap,
                    length=couplingLength if "couplingLength" in dir() else 200.0,
                    fcen=1 / designWvl, df=0.6, nFreq=200, decayTol=1e-3,
                    normalize=True, returnFields=False):
    """
    Core FDTD run: launches a Gaussian source into the straight waveguide and
    records transmitted power at the through port and cross port via flux
    monitors, matching the setup in Fig. 2.
    """
    claddingMed = mp.Medium(index=claddingIndex)
    geometry, yStraight, ySlot = buildCouplerGeometry(slotWidth, gap, length)

    cellX = length + 6.0     # um, +3 um padding each side, not stated, assumed
    cellY = 4.0               # um transverse padding, not stated, assumed
    cell = mp.Vector3(cellX, cellY)

    sourceX = -length / 2 + 1.0
    monitorX = length / 2 - 1.0

    sources = [mp.Source(mp.GaussianSource(frequency=fcen, fwidth=df),
                          component=mp.Ey,
                          center=mp.Vector3(sourceX, yStraight),
                          size=mp.Vector3(0, straightWidth))]

    def makeSim(geom, medium):
        return mp.Simulation(cell_size=cell, resolution=resolutionFDTD,
                              boundary_layers=[mp.PML(pmlThickness)],
                              geometry=geom, default_material=medium,
                              sources=sources)

    # --- reference run (straight waveguide only) for input-power normalisation ---
    refGeom = [mp.Block(size=mp.Vector3(length, straightWidth, mp.inf),
                         center=mp.Vector3(0, yStraight), material=siliconMed2D)]
    simRef = makeSim(refGeom, claddingMed)
    fluxRef = simRef.add_flux(fcen, df, nFreq,
                               mp.FluxRegion(center=mp.Vector3(monitorX, yStraight),
                                             size=mp.Vector3(0, straightWidth)))
    simRef.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ey, mp.Vector3(monitorX, yStraight), decayTol))
    refSpectrum = np.array(mp.get_fluxes(fluxRef))
    freqs = np.array(mp.get_flux_freqs(fluxRef))

    # --- actual coupler run ---
    sim = makeSim(geometry, claddingMed)
    fluxThrough = sim.add_flux(fcen, df, nFreq,
                                mp.FluxRegion(center=mp.Vector3(monitorX, yStraight),
                                              size=mp.Vector3(0, straightWidth)))
    fluxCross = sim.add_flux(fcen, df, nFreq,
                              mp.FluxRegion(center=mp.Vector3(monitorX, ySlot),
                                            size=mp.Vector3(0, slotArmWidth)))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ey, mp.Vector3(monitorX, yStraight), decayTol))

    throughSpectrum = np.array(mp.get_fluxes(fluxThrough))
    crossSpectrum = np.array(mp.get_fluxes(fluxCross))

    if normalize:
        throughSpectrum = throughSpectrum / refSpectrum
        crossSpectrum = crossSpectrum / refSpectrum

    fieldSnapshot = sim.get_array(component=mp.Ey) if returnFields else None
    wavelengths = 1.0 / freqs

    return wavelengths, throughSpectrum, crossSpectrum, fieldSnapshot


couplingLength = 200.0   # um, default coupling length used throughout (paper Fig. 7 caption)


# ==============================================================================
# SECTION 6 : FIGURE 5 - BROADBAND TRANSMISSION SPECTRA
# ==============================================================================
def plotTransmissionSpectra():
    """Recreates Fig. 5: through-port / cross-port power ratio from 1-2 um."""
    fcen = 1 / 1.5   # centred to cover 1-2 um band, per paper wavelength axis
    df = 1.0
    wl, through, cross, _ = runCouplerFDTD(fcen=fcen, df=df, nFreq=400)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(wl, through, label="Through port")
    ax.plot(wl, cross, label="Cross port")
    ax.set_xlim(1.0, 2.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Power ratio")
    ax.set_title("Fig. 5 - Transmission spectra")
    ax.legend()
    fig.savefig(f"{OUT_DIR}/fig5_transmission_spectra.png", dpi=200)
    plt.close(fig)
    return wl, through, cross


# ==============================================================================
# SECTION 7 : FIGURE 6 - CW ELECTRIC-FIELD SNAPSHOTS
# ==============================================================================
def plotFieldSnapshots():
    """Recreates Fig. 6(a)/(b): Ey field at 1550 nm (coupled) and 1200 nm (uncoupled)."""
    shortGap = 0.300      # um, paper caption: reduced separation for visualisation
    shortLength = 15.0    # um, paper caption: shortened coupling length

    fig, axs = plt.subplots(2, 1, figsize=(8, 5))
    for ax, wvl, label in zip(axs, [1.55, 1.20], ["(a) 1550 nm - coupled",
                                                   "(b) 1200 nm - uncoupled"]):
        _, _, _, field = runCouplerFDTD(gap=shortGap, length=shortLength,
                                         fcen=1 / wvl, df=0.05, nFreq=1,
                                         normalize=False, returnFields=True)
        ax.imshow(field.T, cmap="RdBu", origin="lower", aspect="auto")
        ax.set_title(f"Fig. 6 {label}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig6_field_snapshots.png", dpi=200)
    plt.close(fig)


# ==============================================================================
# SECTION 8 : SPECTRAL-DIP HELPER (used by Fig. 7, 8, Table 1, Fig. 9, 10)
# ==============================================================================
def findDipWavelengthAndFWHM(wavelengths, spectrum):
    """Locates the transmission dip wavelength and its FWHM by local interpolation."""
    dipIdx = np.argmin(spectrum)
    dipWvl = wavelengths[dipIdx]
    dipVal = spectrum[dipIdx]
    halfLevel = (1.0 + dipVal) / 2.0

    leftMask = (np.arange(len(spectrum)) < dipIdx) & (spectrum > halfLevel)
    rightMask = (np.arange(len(spectrum)) > dipIdx) & (spectrum > halfLevel)
    leftWvl = wavelengths[leftMask][-1] if leftMask.any() else wavelengths[0]
    rightWvl = wavelengths[rightMask][0] if rightMask.any() else wavelengths[-1]
    fwhm = abs(rightWvl - leftWvl)
    return dipWvl, fwhm


# ==============================================================================
# SECTION 9 : FIGURES 7 & 8 - SENSITIVITY TO CLADDING REFRACTIVE INDEX
# ==============================================================================
def plotSensitivity():
    """Recreates Fig. 7 (two-sample shift) and Fig. 8 (sensitivity linear fit)."""
    fcen, df = 1 / 1.6, 0.3   # narrow band around the ~1590 nm coupling feature

    # --- Fig. 7: two-sample dip shift, cladding 1.318 vs 1.34 ---
    fig7, ax7 = plt.subplots(figsize=(6, 4))
    for n, label in [(1.318, "sample 1: 1.318"), (1.34, "sample 2: 1.34")]:
        wl, through, _, _ = runCouplerFDTD(claddingIndex=n, fcen=fcen, df=df, nFreq=300)
        ax7.plot(wl, through, label=label)
    ax7.set_xlabel("Wavelength (um)")
    ax7.set_ylabel("Power ratio")
    ax7.set_title("Fig. 7 - Through-port shift for two cladding samples")
    ax7.legend()
    fig7.savefig(f"{OUT_DIR}/fig7_sample_shift.png", dpi=200)
    plt.close(fig7)

    # --- Fig. 8: sweep cladding index 1.318 to 1.400, fit sensitivity ---
    riRange = np.linspace(1.318, 1.40, 9)    # 9 points, matches paper's plotted range
    dipWvls = []
    for n in riRange:
        wl, through, _, _ = runCouplerFDTD(claddingIndex=n, fcen=fcen, df=df, nFreq=300)
        dipWvl, _ = findDipWavelengthAndFWHM(wl, through)
        dipWvls.append(dipWvl)
    dipWvls = np.array(dipWvls)

    slope, intercept = np.polyfit(riRange, dipWvls, 1)
    sensitivity_nm_per_RIU = slope * 1000   # um -> nm

    fig8, ax8 = plt.subplots(figsize=(6, 4))
    ax8.plot(riRange, dipWvls, "o", label="Data points")
    ax8.plot(riRange, slope * riRange + intercept, "--",
             label=f"Best fit (Sensitivity = {sensitivity_nm_per_RIU:.2f} nm/RIU)")
    ax8.set_xlabel("Refractive index")
    ax8.set_ylabel("Coupled wavelength (um)")
    ax8.set_title("Fig. 8 - Sensitivity plot")
    ax8.legend()
    fig8.savefig(f"{OUT_DIR}/fig8_sensitivity.png", dpi=200)
    plt.close(fig8)

    return sensitivity_nm_per_RIU


# ==============================================================================
# SECTION 10 : TABLE 1 - GAP-WIDTH SWEEP (SENSITIVITY, FWHM, FOM)
# ==============================================================================
def searchOptimalLength(gap, slotWidth=slotGapWidth, lengthGuesses=None):
    """Finds the coupling length that maximises cross-port power for a given gap."""
    if lengthGuesses is None:
        lengthGuesses = np.linspace(100, 400, 7)   # um, search range not given in paper
    bestLength, bestCross = lengthGuesses[0], -1
    for L in lengthGuesses:
        wl, _, cross, _ = runCouplerFDTD(gap=gap, slotWidth=slotWidth, length=L,
                                          fcen=1 / 1.6, df=0.3, nFreq=100)
        peakCross = cross.max()
        if peakCross > bestCross:
            bestCross, bestLength = peakCross, L
    return bestLength


def buildGapWidthTable():
    """Recreates Table 1: sensitivity, FWHM, and FOM vs coupling-gap width."""
    gapValues = [0.400, 0.500, 0.600, 0.700]   # um
    riRange = np.linspace(1.318, 1.40, 5)
    rows = []
    for gap in gapValues:
        optLength = searchOptimalLength(gap)
        dipWvls, fwhms = [], []
        for n in riRange:
            wl, through, _, _ = runCouplerFDTD(claddingIndex=n, gap=gap, length=optLength,
                                                fcen=1 / 1.6, df=0.3, nFreq=300)
            dipWvl, fwhm = findDipWavelengthAndFWHM(wl, through)
            dipWvls.append(dipWvl)
            fwhms.append(fwhm)
        slope, _ = np.polyfit(riRange, dipWvls, 1)
        sensitivity = slope * 1000            # nm/RIU
        avgFWHM = np.mean(fwhms) * 1000        # nm
        fom = sensitivity / avgFWHM
        rows.append((gap * 1000, sensitivity, avgFWHM, fom))

    print("Gap (nm) | Sensitivity (nm/RIU) | Avg FWHM (nm) | FOM (1/RIU)")
    for gapNm, sens, fwhm, fom in rows:
        print(f"{gapNm:8.0f} | {sens:20.1f} | {fwhm:13.1f} | {fom:11.2f}")
    return rows


# ==============================================================================
# SECTION 11 : FIGURE 9 - SLOT-WIDTH SWEEP (ON-CHIP SPECTROMETER CHANNELS)
# ==============================================================================
def plotSlotWidthSweep():
    """Recreates Fig. 9: coupled wavelength vs slot width, silica-clad device."""
    silicaClad = nOxide      # silica cladding replaces water for the spectrometer variant
    slotWidths = np.arange(0.050, 0.061, 0.001)   # 50-60 nm, 1 nm steps
    dipWvls = []
    for w in slotWidths:
        wl, through, _, _ = runCouplerFDTD(claddingIndex=silicaClad, slotWidth=w,
                                            fcen=1 / 1.6, df=0.3, nFreq=300)
        dipWvl, _ = findDipWavelengthAndFWHM(wl, through)
        dipWvls.append(dipWvl)
    dipWvls = np.array(dipWvls)

    slope, intercept = np.polyfit(slotWidths * 1000, dipWvls, 1)   # nm -> um/nm slope

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(slotWidths * 1000, dipWvls, "o--",
            label=f"Best fit (Sensitivity = {slope:.2f} nm/nm)")
    ax.set_xlabel("Slot width (nm)")
    ax.set_ylabel("Coupled peak wavelength (um)")
    ax.set_title("Fig. 9 - Coupled wavelength vs slot width")
    ax.legend()
    fig.savefig(f"{OUT_DIR}/fig9_slot_width_sweep.png", dpi=200)
    plt.close(fig)


# ==============================================================================
# SECTION 12 : FIGURE 10 - THERMAL TUNING SWEEP
# ==============================================================================
def siliconIndexAtTemperature(temperatureK, refTemperature=293.0,
                               thermoOpticCoeff=1.8e-4):
    """Silicon index vs temperature via its thermo-optic coefficient (paper Sec. 3)."""
    return nSilicon + thermoOpticCoeff * (temperatureK - refTemperature)


def runThermalCoupler(temperatureK, **kwargs):
    """Runs the coupler with a temperature-shifted silicon index."""
    nSiT = siliconIndexAtTemperature(temperatureK)
    effIndexSiT = effIndexSi + (nSiT - nSilicon)   # shift effective index by the same delta
    global siliconMed2D
    originalMed = siliconMed2D
    siliconMed2D = mp.Medium(index=effIndexSiT)
    try:
        result = runCouplerFDTD(claddingIndex=nOxide, **kwargs)  # silica-clad spectrometer variant
    finally:
        siliconMed2D = originalMed
    return result


def plotThermalTuning():
    """Recreates Fig. 10: coupled wavelength vs temperature, with 3 K fine-step inset."""
    broadTemps = np.linspace(293, 600, 8)     # K, broad sweep (paper: 293-600 K)
    fineTemps = [303, 306, 309]                 # K, fine 3 K steps (paper resolution test)

    def sweepDips(temps):
        dips = []
        for T in temps:
            wl, through, _, _ = runThermalCoupler(T, fcen=1 / 1.6, df=0.3, nFreq=300)
            dipWvl, _ = findDipWavelengthAndFWHM(wl, through)
            dips.append(dipWvl)
        return np.array(dips)

    broadDips = sweepDips(broadTemps)
    fineDips = sweepDips(fineTemps)

    slopeBroad, _ = np.polyfit(broadTemps, broadDips * 1000, 1)   # nm/K
    slopeFine, _ = np.polyfit(fineTemps, fineDips * 1000, 1)      # nm/K

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(broadTemps, broadDips, "o--",
            label=f"Best fit (Temperature sensitivity = {slopeBroad:.2f} nm/K)")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Coupled wavelength (um)")
    ax.set_title("Fig. 10 - Thermal tuning")
    ax.legend()

    insetAx = fig.add_axes([0.58, 0.2, 0.3, 0.3])
    insetAx.plot(fineTemps, fineDips, "o--")
    insetAx.set_title(f"{slopeFine:.2f} nm/K (3 K steps)", fontsize=7)

    fig.savefig(f"{OUT_DIR}/fig10_thermal_tuning.png", dpi=200)
    plt.close(fig)


# ==============================================================================
# SECTION 13 : MAIN - RUN ALL REPRODUCTIONS
# ==============================================================================
if __name__ == "__main__":
    plotCrossSection()
    plotSimulationSchematic()
    plotModeDispersion()
    plotTransmissionSpectra()
    plotFieldSnapshots()
    plotSensitivity()
    buildGapWidthTable()
    plotSlotWidthSweep()
    plotThermalTuning()
    print(f"All figures written to ./{OUT_DIR}/")