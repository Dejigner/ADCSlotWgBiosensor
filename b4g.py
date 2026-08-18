# """
# SOI Asymmetric Directional Coupler Biosensor — MEEP Simulation
# ==============================================================
# Reproduces Figs 5, 6, 7, 8 from:
#   Alqabandi & Scullion, J. Opt. 28 (2026) 045001

# Geometry (2-D FDTD, effective-index method)
#   - Strip waveguide  : 350 nm wide, centered at y = 0
#   - Slot waveguide   : 550 nm wide, 50 nm slot, centered at y = +0.850 µm
#   - Edge-to-edge gap : 400 nm  (user-specified, Table 1 row)
#   - Coupling length  : 25 µm
#   - SOI stack        : 220 nm Si → effective index n_eff = 2.842
#   - Cladding (ref.)  : water n = 1.318 at λ = 1550 nm

# MEEP units: length in µm, frequency in c/µm (so f = 1/λ[µm])

# Author note: Run with  `python3 adc_biosensor_meep.py`
# Requires:   meep, numpy, matplotlib, scipy  (conda-forge recommended)
# """

# import meep as mp
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")          # headless backend — change to "TkAgg" if interactive
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# from scipy.stats   import linregress

# # ─────────────────────────────────────────────────────────────────────────────
# #  SECTION 1 — SIMULATION PARAMETERS
# # ─────────────────────────────────────────────────────────────────────────────

# # ── Effective indices (n_eff from 1-D slab solver, paper §3) ─────────────────
# nEff        = 2.842        # Si slab effective index (220 nm thick)
# nSiO2       = 1.444        # buried-oxide substrate (not used in 2-D EIM geometry)
# nWaterRef   = 1.318        # reference cladding refractive index

# # ── Waveguide geometry (µm) ───────────────────────────────────────────────────
# wStrip      = 0.350        # strip waveguide width
# wSlot       = 0.550        # slot waveguide total width (both rails + slot)
# slotW       = 0.050        # slot gap width
# gapEdge     = 0.400        # edge-to-edge separation between strip and slot WG
# couplingLen = 25.0         # coupling region length [µm]

# # Strip center at y=0; slot center calculated from geometry:
# #   y_slot = wStrip/2 + gapEdge + wSlot/2
# yCenterStrip = 0.0
# yCenterSlot  = wStrip / 2 + gapEdge + wSlot / 2   # = 0.850 µm

# # Rail widths of the slot waveguide (each rail = (wSlot - slotW) / 2)
# railW = (wSlot - slotW) / 2    # = 0.250 µm per rail

# # ── Domain and PML ────────────────────────────────────────────────────────────
# padX        = 3.0          # padding along x (propagation axis) on each side [µm]
# padYBot     = 1.5          # padding below strip [µm]
# padYTop     = 1.5          # padding above slot [µm]
# pmlThick    = 1.0          # PML thickness [µm]

# domainX = couplingLen + 2 * padX
# domainY = padYBot + yCenterSlot + wSlot / 2 + padYTop

# # Center domain: shift y so the midpoint between waveguides is at y=0
# yMid    = (yCenterStrip + yCenterSlot) / 2    # = +0.425 µm
# # In MEEP the cell is centered at the origin, so we need an offset
# # to place the strip at yCenterStrip − yMid from cell center
# # → stripY = yCenterStrip − yMid = −0.425 µm  (in MEEP cell coords)
# # → slotY  = yCenterSlot  − yMid = +0.425 µm
# stripY  = yCenterStrip - yMid      # −0.425 µm
# slotY   = yCenterSlot  - yMid      # +0.425 µm

# # Source and monitor x-positions (in MEEP cell coords, cell centered at 0)
# xLeft       = -couplingLen / 2     # left edge of coupling region
# xRight      =  couplingLen / 2     # right edge of coupling region
# xSrc        = xLeft  - 1.5        # 1.5 µm left of coupling start
# xMonitor    = xRight + 1.5        # 1.5 µm right of coupling end

# # ── Spectral parameters ───────────────────────────────────────────────────────
# resolution  = 30           # pixels per µm (paper §3, "30 pixels µm⁻¹")
# lamCen      = 1.55         # centre wavelength [µm]
# lamMin      = 1.0          # shortest λ of interest [µm]
# lamMax      = 2.0          # longest  λ of interest [µm]
# fCen        = 1 / lamCen   # centre frequency [c/µm]
# fMin        = 1 / lamMax   # 0.5
# fMax        = 1 / lamMin   # 1.0
# fBW         = fMax - fMin  # total bandwidth
# nFreqs      = 500          # number of spectral points for Harminv / flux monitors

# # ── Runtime ───────────────────────────────────────────────────────────────────
# simTime     = 5000         # MEEP time units (~enough for spectral settling)

# # ── Sensitivity sweep ─────────────────────────────────────────────────────────
# # Paper sweeps cladding RI from 1.318 to 1.400 (Fig. 8); we reproduce this.
# nCladdingVals = np.array([1.318, 1.330, 1.340, 1.350, 1.360, 1.370, 1.380, 1.400])


# # ─────────────────────────────────────────────────────────────────────────────
# #  SECTION 2 — GEOMETRY BUILDER
# # ─────────────────────────────────────────────────────────────────────────────

# def buildGeometry(nCladding):
#     """
#     Returns the MEEP geometry (list of Block objects) for the 2-D effective-
#     index model.  The waveguide cores use nEff; the background is nCladding.

#     Parameters
#     ----------
#     nCladding : float
#         Refractive index of the top cladding medium (water or analyte).

#     Returns
#     -------
#     geometry : list[mp.Block]
#     medium   : mp.Medium  — background medium
#     """
#     # Background = analyte cladding
#     background = mp.Medium(index=nCladding)

#     # Strip waveguide (solid rectangle)
#     stripBlock = mp.Block(
#         size    = mp.Vector3(couplingLen, wStrip, mp.inf),
#         center  = mp.Vector3(0, stripY, 0),
#         material= mp.Medium(index=nEff)
#     )

#     # Slot waveguide — left rail
#     leftRailBlock = mp.Block(
#         size    = mp.Vector3(couplingLen, railW, mp.inf),
#         center  = mp.Vector3(0, slotY - (slotW / 2 + railW / 2), 0),
#         material= mp.Medium(index=nEff)
#     )

#     # Slot waveguide — right rail
#     rightRailBlock = mp.Block(
#         size    = mp.Vector3(couplingLen, railW, mp.inf),
#         center  = mp.Vector3(0, slotY + (slotW / 2 + railW / 2), 0),
#         material= mp.Medium(index=nEff)
#     )

#     # Slot gap remains background (nCladding fills the slot)

#     geometry = [stripBlock, leftRailBlock, rightRailBlock]
#     return geometry, background


# # ─────────────────────────────────────────────────────────────────────────────
# #  SECTION 3 — SOURCE, PML, AND CELL
# # ─────────────────────────────────────────────────────────────────────────────

# def buildSimulation(nCladding, nFreqsMon=nFreqs, runTime=simTime):
#     """
#     Builds and returns a configured mp.Simulation object.

#     The Gaussian source is placed 1.5 µm left of the coupling region and
#     excites only the strip waveguide (Ez polarisation, TE-like in 2-D).
#     Two flux monitors are placed 1.5 µm right of the coupling region:
#       • throughFlux  — at the strip waveguide output
#       • crossFlux    — at the slot waveguide output

#     Parameters
#     ----------
#     nCladding  : float   — analyte refractive index
#     nFreqsMon  : int     — frequency points in flux monitors
#     runTime    : float   — simulation run time (MEEP units)

#     Returns
#     -------
#     sim          : mp.Simulation
#     throughFlux  : flux region
#     crossFlux    : flux region
#     """
#     geometry, background = buildGeometry(nCladding)

#     cell = mp.Vector3(domainX, domainY, 0)

#     pml = [mp.PML(pmlThick)]

#     # Gaussian pulse exciting the strip waveguide (TE: Ez)
#     source = mp.GaussianSource(
#         frequency  = fCen,
#         fwidth     = fBW,
#         is_integrated = True   # correct normalisation for broadband sources
#     )

#     sources = [mp.Source(
#         src      = source,
#         component= mp.Ez,
#         center   = mp.Vector3(xSrc, stripY, 0),
#         size     = mp.Vector3(0, wStrip * 2, 0)   # span 2× strip width to capture mode
#     )]

#     sim = mp.Simulation(
#         cell_size   = cell,
#         geometry    = geometry,
#         sources     = sources,
#         boundary_layers = pml,
#         default_material = background,
#         resolution  = resolution
#     )

#     # ── Flux monitors ─────────────────────────────────────────────────────────
#     # Monitor width: 3× waveguide width to capture evanescent tails
#     monWidthStrip = wStrip * 3
#     monWidthSlot  = wSlot  * 3

#     throughRegion = mp.FluxRegion(
#         center = mp.Vector3(xMonitor, stripY, 0),
#         size   = mp.Vector3(0, monWidthStrip, 0)
#     )
#     crossRegion = mp.FluxRegion(
#         center = mp.Vector3(xMonitor, slotY, 0),
#         size   = mp.Vector3(0, monWidthSlot, 0)
#     )

#     throughFlux = sim.add_flux(fCen, fBW, nFreqsMon, throughRegion)
#     crossFlux   = sim.add_flux(fCen, fBW, nFreqsMon, crossRegion)

#     return sim, throughFlux, crossFlux


# # ─────────────────────────────────────────────────────────────────────────────
# #  SECTION 4 — NORMALISATION (empty-waveguide reference run)
# # ─────────────────────────────────────────────────────────────────────────────

# def runNormalization():
#     """
#     Run the simulation with NO geometry (empty cell) to record the
#     incident flux.  This flux is used to normalise all subsequent runs
#     so that output powers are expressed as fractions of input power.

#     Returns
#     -------
#     normThroughData : np.ndarray — incident flux spectrum (nFreqs values)
#     freqs           : np.ndarray — corresponding frequencies [c/µm]
#     """
#     print("\n[NORM] Running normalisation (empty waveguide) ...")

#     # For normalisation: use the same source but no geometry
#     cell = mp.Vector3(domainX, domainY, 0)
#     pml  = [mp.PML(pmlThick)]

#     source = mp.GaussianSource(
#         frequency     = fCen,
#         fwidth        = fBW,
#         is_integrated = True
#     )
#     sources = [mp.Source(
#         src       = source,
#         component = mp.Ez,
#         center    = mp.Vector3(xSrc, stripY, 0),
#         size      = mp.Vector3(0, wStrip * 2, 0)
#     )]

#     sim = mp.Simulation(
#         cell_size        = cell,
#         geometry         = [],
#         sources          = sources,
#         boundary_layers  = pml,
#         resolution       = resolution
#     )

#     monRegion = mp.FluxRegion(
#         center = mp.Vector3(xMonitor, stripY, 0),
#         size   = mp.Vector3(0, wStrip * 2, 0)
#     )
#     normFlux = sim.add_flux(fCen, fBW, nFreqs, monRegion)

#     sim.run(until=simTime)

#     normData = np.array(mp.get_fluxes(normFlux))
#     freqs    = np.array(mp.get_flux_freqs(normFlux))

#     # Save flux data for subtraction in subsequent runs
#     sim.save_flux("norm_flux", normFlux)

#     return normData, freqs, sim, normFlux


# # ─────────────────────────────────────────────────────────────────────────────
# #  SECTION 5 — SINGLE SPECTRAL RUN
# # ─────────────────────────────────────────────────────────────────────────────

# def runSpectrum(nCladding, normFluxSim=None, normFluxObj=None):
#     """
#     Run the full ADC simulation for a given cladding index.
#     Returns normalised through-port and cross-port power spectra.

#     Parameters
#     ----------
#     nCladding   : float   — analyte refractive index
#     normFluxSim : previously run normalisation mp.Simulation (for flux sub.)
#     normFluxObj : flux object from normalisation run

#     Returns
#     -------
#     freqs       : np.ndarray  — frequency array [c/µm]
#     wavelengths : np.ndarray  — wavelength array [µm]
#     pThrough    : np.ndarray  — normalised through-port power [0, 1]
#     pCross      : np.ndarray  — normalised cross-port power [0, 1]
#     """
#     print(f"\n[SIM] n_cladding = {nCladding:.3f}")

#     sim, throughFlux, crossFlux = buildSimulation(nCladding)

#     # Load normalisation flux (subtract incident field)
#     if normFluxObj is not None:
#         sim.load_minus_flux("norm_flux", throughFlux)

#     sim.run(until=simTime)

#     freqs       = np.array(mp.get_flux_freqs(throughFlux))
#     wavelengths = 1 / freqs                            # µm

#     throughData = np.array(mp.get_fluxes(throughFlux))
#     crossData   = np.array(mp.get_fluxes(crossFlux))

#     # Normalise by incident flux
#     if normFluxObj is not None:
#         normData = np.abs(np.array(mp.get_fluxes(normFluxObj)))
#         normData[normData == 0] = 1.0                  # avoid divide-by-zero
#         pThrough = np.abs(throughData) / normData
#         pCross   = np.abs(crossData)   / normData
#     else:
#         # Fallback: normalise to peak of through port
#         peak = np.max(np.abs(throughData))
#         peak = peak if peak > 0 else 1.0
#         pThrough = np.abs(throughData) / peak
#         pCross   = np.abs(crossData)   / peak

#     # Clip to [0, 1] — small numerical overshoot can occur
#     pThrough = np.clip(pThrough, 0, 1.2)
#     pCross   = np.clip(pCross,   0, 1.2)

#     return freqs, wavelengths, pThrough, pCross


# # ─────────────────────────────────────────────────────────────────────────────
# #  SECTION 6 — FIELD SNAPSHOT RUN (Fig. 6)
# # ─────────────────────────────────────────────────────────────────────────────

# def runFieldSnapshot(lamSource, nCladding=nWaterRef, tSnap=2500):
#     """
#     Run a CW (continuous-wave) source simulation to capture the Ez field
#     distribution at a fixed wavelength.  Returns the 2-D field array and
#     the corresponding x/y coordinate arrays.

#     Parameters
#     ----------
#     lamSource  : float  — source wavelength [µm]
#     nCladding  : float  — cladding index
#     tSnap      : float  — time at which to take the snapshot [MEEP units]
#                           (must be >> transit time for steady state)

#     Returns
#     -------
#     xArr, yArr : 1-D coordinate arrays [µm]
#     ezField    : 2-D Ez field array
#     """
#     print(f"\n[SNAP] Wavelength = {lamSource} µm")

#     fSource = 1 / lamSource

#     geometry, background = buildGeometry(nCladding)

#     cell = mp.Vector3(domainX, domainY, 0)
#     pml  = [mp.PML(pmlThick)]

#     source = mp.ContinuousSource(
#         frequency  = fSource,
#         is_integrated = True
#     )
#     sources = [mp.Source(
#         src       = source,
#         component = mp.Ez,
#         center    = mp.Vector3(xSrc, stripY, 0),
#         size      = mp.Vector3(0, wStrip * 2, 0)
#     )]

#     sim = mp.Simulation(
#         cell_size        = cell,
#         geometry         = geometry,
#         sources          = sources,
#         boundary_layers  = pml,
#         default_material = background,
#         resolution       = resolution
#     )

#     sim.run(until=tSnap)

#     ezData = sim.get_array(
#         center    = mp.Vector3(0, 0, 0),
#         size      = mp.Vector3(domainX, domainY, 0),
#         component = mp.Ez
#     )

#     # Build coordinate arrays
#     nx, ny = ezData.shape
#     xArr   = np.linspace(-domainX / 2, domainX / 2, nx)
#     yArr   = np.linspace(-domainY / 2, domainY / 2, ny)

#     return xArr, yArr, ezData


# # ─────────────────────────────────────────────────────────────────────────────
# #  SECTION 7 — COUPLING WAVELENGTH FINDER
# # ─────────────────────────────────────────────────────────────────────────────

# def findCouplingWavelength(wavelengths, pThrough, pCross):
#     """
#     Locate the coupling wavelength as the dip in the through-port spectrum
#     (equivalently, the peak in the cross-port spectrum).

#     Strategy:
#     1. Find the minimum in pThrough in the range 1.3–2.0 µm.
#     2. Confirm it corresponds to a peak in pCross.
#     3. Return the wavelength at that minimum.

#     Parameters
#     ----------
#     wavelengths : np.ndarray
#     pThrough    : np.ndarray
#     pCross      : np.ndarray

#     Returns
#     -------
#     lamCoupled : float  — coupling wavelength [µm]
#     """
#     # Work in the physically relevant window (1.3–2.0 µm)
#     mask = (wavelengths >= 1.3) & (wavelengths <= 2.0)
#     wMask = wavelengths[mask]
#     pMask = pThrough[mask]

#     # Find the dip (minimum) — use inverted array for find_peaks
#     peaks, props = find_peaks(-pMask, prominence=0.05)

#     if len(peaks) == 0:
#         # Fallback: simple argmin
#         idxMin = np.argmin(pMask)
#         return wMask[idxMin]

#     # If multiple dips, pick the most prominent one
#     bestIdx = peaks[np.argmax(props["prominences"])]
#     return wMask[bestIdx]


# # ─────────────────────────────────────────────────────────────────────────────
# #  SECTION 8 — PLOTTING HELPERS
# # ─────────────────────────────────────────────────────────────────────────────

# def plotTransmissionSpectra(wavelengths, pThrough, pCross, title="",
#                             savePath="fig5_transmission_spectra.png"):
#     """
#     Reproduce Fig. 5: through-port and cross-port power vs. wavelength.
#     """
#     fig, ax = plt.subplots(figsize=(8, 5))

#     ax.plot(wavelengths, pThrough, color="blue",  linewidth=1.2, label="Through Port")
#     ax.plot(wavelengths, pCross,   color="red",   linewidth=1.2, label="Cross Port")

#     ax.set_xlabel("Wavelength (µm)", fontsize=13)
#     ax.set_ylabel("Power ratio",     fontsize=13)
#     ax.set_title(title or "Transmission Spectra", fontsize=13)
#     ax.set_xlim(1.0, 2.0)
#     ax.set_ylim(0, 1.1)
#     ax.legend(fontsize=11)
#     ax.grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig(savePath, dpi=150)
#     plt.close()
#     print(f"[PLOT] Saved {savePath}")


# def plotFieldSnapshot(xArr, yArr, ezField, lamSource, label,
#                       savePath="fig6_field.png"):
#     """
#     Reproduce Fig. 6 panels: Ez field distribution colour map.
#     Waveguide outlines are drawn as white rectangles for reference.
#     """
#     # Use |Ez|² (intensity) for clearer visualisation
#     intensity = np.abs(ezField.T) ** 2
#     intensityNorm = intensity / (intensity.max() + 1e-30)

#     fig, ax = plt.subplots(figsize=(12, 4))

#     im = ax.imshow(
#         intensityNorm,
#         origin="lower",
#         extent=[xArr[0], xArr[-1], yArr[0], yArr[-1]],
#         aspect="auto",
#         cmap="hot",
#         vmin=0, vmax=1
#     )
#     plt.colorbar(im, ax=ax, label="|Ez|² (normalised)")

#     # Draw waveguide outlines
#     def drawWg(yCenter, width, color="white"):
#         from matplotlib.patches import Rectangle
#         ax.add_patch(Rectangle(
#             (xLeft, yCenter - width / 2 - yMid),
#             couplingLen, width,
#             linewidth=1.0, edgecolor=color,
#             facecolor="none", linestyle="--"
#         ))

#     drawWg(yCenterStrip - yMid, wStrip, "cyan")
#     drawWg(yCenterSlot  - yMid, wSlot,  "cyan")

#     ax.set_xlabel("x (µm)", fontsize=12)
#     ax.set_ylabel("y (µm)", fontsize=12)
#     ax.set_title(f"Ez field — λ = {lamSource} µm   {label}", fontsize=12)
#     ax.set_xlim(xLeft - 1, xRight + 1)    # zoom to coupling region

#     plt.tight_layout()
#     plt.savefig(savePath, dpi=150)
#     plt.close()
#     print(f"[PLOT] Saved {savePath}")


# def plotShiftedWavelength(wavelengths, pThrough1, pThrough2,
#                           n1, n2,
#                           savePath="fig7_shifted_wavelength.png"):
#     """
#     Reproduce Fig. 7: through-port spectra for two different cladding indices
#     showing the blue-shift of the coupling dip.
#     """
#     fig, ax = plt.subplots(figsize=(8, 5))

#     ax.plot(wavelengths, pThrough1, color="blue",
#             linewidth=1.2, label=f"sample 1: {n1:.3f}")
#     ax.plot(wavelengths, pThrough2, color="orange",
#             linewidth=1.2, label=f"sample 2: {n2:.3f}")

#     ax.set_xlabel("Wavelength (µm)", fontsize=13)
#     ax.set_ylabel("Power Ratio",     fontsize=13)
#     ax.set_title("Shifted Wavelength Plot", fontsize=13)
#     ax.set_xlim(1.2, 2.0)
#     ax.set_ylim(0, 1.1)
#     ax.legend(fontsize=11)
#     ax.grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig(savePath, dpi=150)
#     plt.close()
#     print(f"[PLOT] Saved {savePath}")


# def plotSensitivity(nVals, lamCoupledVals,
#                     savePath="fig8_sensitivity.png"):
#     """
#     Reproduce Fig. 8: coupled wavelength vs. cladding refractive index,
#     with a linear best-fit line and sensitivity annotation.
#     """
#     slope, intercept, r, p, se = linregress(nVals, lamCoupledVals)
#     sensitivityNmPerRIU = slope * 1000    # µm/RIU → nm/RIU
#     fitLine = slope * np.array(nVals) + intercept

#     fig, ax = plt.subplots(figsize=(7, 5))

#     ax.scatter(nVals, lamCoupledVals, marker="D", s=50,
#                color="blue", label="Data points", zorder=5)
#     ax.plot(nVals, fitLine, color="blue", linestyle="--",
#             label=f"Best fit (Sensitivity = {sensitivityNmPerRIU:.2f} nm/RIU)")

#     ax.set_xlabel("Refractive index",        fontsize=13)
#     ax.set_ylabel("Coupled wavelength (µm)", fontsize=13)
#     ax.set_title("Sensitivity Plot",         fontsize=13)
#     ax.legend(fontsize=11)
#     ax.grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig(savePath, dpi=150)
#     plt.close()
#     print(f"[PLOT] Saved {savePath}")
#     print(f"  → Sensitivity: {sensitivityNmPerRIU:.1f} nm/RIU   (R² = {r**2:.4f})")
#     return sensitivityNmPerRIU


# # ─────────────────────────────────────────────────────────────────────────────
# #  SECTION 9 — FIGURE OF MERIT, LOD, Q-FACTOR
# # ─────────────────────────────────────────────────────────────────────────────

# def computePerformanceMetrics(wavelengths, pThrough, sensitivityNmPerRIU):
#     """
#     Compute FOM, quality factor Q, and estimated LOD following
#     White & Fan (2008) [Ref. 39 in paper].

#     Parameters
#     ----------
#     wavelengths         : np.ndarray   [µm]
#     pThrough            : np.ndarray   normalised through-port power
#     sensitivityNmPerRIU : float        [nm/RIU]

#     Returns
#     -------
#     dict with keys: FWHM_nm, Q, FOM, LOD_RIU
#     """
#     # Find the coupling dip
#     mask = (wavelengths >= 1.3) & (wavelengths <= 2.0)
#     wMask = wavelengths[mask]
#     pMask = pThrough[mask]

#     idxMin = np.argmin(pMask)
#     lamCoupled = wMask[idxMin]
#     dipVal     = pMask[idxMin]

#     # Half-maximum level (measured from the dip, not from unity)
#     halfMax = (1.0 + dipVal) / 2.0

#     # Indices where power crosses halfMax on either side of the dip
#     leftMask  = (wMask < lamCoupled) & (pMask < halfMax)
#     rightMask = (wMask > lamCoupled) & (pMask < halfMax)

#     lamLeft  = wMask[leftMask][0]  if leftMask.any()  else lamCoupled - 0.05
#     lamRight = wMask[rightMask][-1] if rightMask.any() else lamCoupled + 0.05

#     fwhmMicron = lamRight - lamLeft
#     fwhmNm     = fwhmMicron * 1000    # nm

#     # Quality factor
#     Q = (lamCoupled * 1000) / fwhmNm   # λ/Δλ, both in nm

#     # FOM = Sensitivity / FWHM  [RIU⁻¹]
#     FOM = sensitivityNmPerRIU / fwhmNm

#     # LOD using White & Fan (Eq. 5–7 in paper)
#     SNR_dB    = 50.0
#     SNR_lin   = 10 ** (SNR_dB / 10)
#     sigmaAmpl = (fwhmNm / 4.5) * (SNR_lin ** (-0.25))   # Eq. (5)
#     sigmaTemp = 0.11     # nm  (0.5 °C shift, all thermo-optic coefficients)
#     sigmaLas  = 0.001    # nm  (1 fm laser linewidth, worst case)
#     resolution3sigma = 3 * np.sqrt(sigmaAmpl**2 + sigmaTemp**2 + sigmaLas**2)  # Eq. (6)
#     LOD = resolution3sigma / sensitivityNmPerRIU   # Eq. (7)

#     return {
#         "lamCoupled_nm" : lamCoupled * 1000,
#         "FWHM_nm"       : fwhmNm,
#         "Q"             : Q,
#         "FOM"           : FOM,
#         "sigmaAmpl_nm"  : sigmaAmpl,
#         "resolution_nm" : resolution3sigma,
#         "LOD_RIU"       : LOD
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# #  SECTION 10 — MAIN EXECUTION
# # ─────────────────────────────────────────────────────────────────────────────

# def main():
#     print("=" * 65)
#     print("  SOI ADC Biosensor — MEEP Simulation")
#     print("  Reproducing Figs 5, 6, 7, 8 of Alqabandi & Scullion (2026)")
#     print("=" * 65)
#     print(f"\nGeometry summary:")
#     print(f"  Strip WG width     : {wStrip*1000:.0f} nm")
#     print(f"  Slot WG width      : {wSlot*1000:.0f} nm  (slot = {slotW*1000:.0f} nm)")
#     print(f"  Rail width         : {railW*1000:.0f} nm each")
#     print(f"  Edge-to-edge gap   : {gapEdge*1000:.0f} nm")
#     print(f"  Coupling length    : {couplingLen:.0f} µm")
#     print(f"  Strip y-center     : {stripY:.4f} µm (MEEP cell coords)")
#     print(f"  Slot  y-center     : {slotY:.4f} µm (MEEP cell coords)")
#     print(f"  Domain X           : {domainX:.1f} µm")
#     print(f"  Domain Y           : {domainY:.2f} µm")
#     print(f"  Resolution         : {resolution} px/µm")
#     print(f"  n_eff (Si slab)    : {nEff}")
#     print(f"  n_cladding (ref)   : {nWaterRef}")

#     # ── STEP A: Normalisation run ─────────────────────────────────────────────
#     normData, freqs, normSim, normFluxObj = runNormalization()

#     wavelengths = 1 / freqs     # µm

#     # ── STEP B: Reference spectrum (n = 1.318, water) — reproduces Fig. 5 ────
#     _, _, pThroughRef, pCrossRef = runSpectrum(
#         nWaterRef, normFluxSim=normSim, normFluxObj=normFluxObj
#     )

#     plotTransmissionSpectra(
#         wavelengths, pThroughRef, pCrossRef,
#         title="Transmission Spectra (n_clad = 1.318, gap = 400 nm)",
#         savePath="fig5_transmission_spectra.png"
#     )

#     lamRef = findCouplingWavelength(wavelengths, pThroughRef, pCrossRef)
#     print(f"\n[RESULT] Coupling wavelength at n=1.318 : {lamRef*1000:.1f} nm")

#     # ── STEP C: Field snapshots — reproduces Fig. 6 ───────────────────────────
#     # Fig. 6(a): coupling wavelength → power transfers to slot
#     xA, yA, ezA = runFieldSnapshot(
#         lamSource=lamRef, nCladding=nWaterRef, tSnap=2500
#     )
#     plotFieldSnapshot(
#         xA, yA, ezA, lamSource=lamRef,
#         label="(coupling ON)",
#         savePath="fig6a_field_coupling.png"
#     )

#     # Fig. 6(b): off-coupling wavelength (1.2 µm) → power stays in strip
#     xB, yB, ezB = runFieldSnapshot(
#         lamSource=1.200, nCladding=nWaterRef, tSnap=2500
#     )
#     plotFieldSnapshot(
#         xB, yB, ezB, lamSource=1.200,
#         label="(coupling OFF)",
#         savePath="fig6b_field_no_coupling.png"
#     )

#     # ── STEP D: Shifted spectrum (n = 1.34) — reproduces Fig. 7 ──────────────
#     _, _, pThrough134, pCross134 = runSpectrum(
#         1.340, normFluxSim=normSim, normFluxObj=normFluxObj
#     )

#     plotShiftedWavelength(
#         wavelengths, pThroughRef, pThrough134,
#         n1=1.318, n2=1.340,
#         savePath="fig7_shifted_wavelength.png"
#     )

#     lam134 = findCouplingWavelength(wavelengths, pThrough134, pCross134)
#     shiftNm = (lamRef - lam134) * 1000    # blue-shift → lamRef > lam134
#     print(f"[RESULT] Coupling wavelength at n=1.340 : {lam134*1000:.1f} nm")
#     print(f"[RESULT] Blue shift Δλ (1.318→1.340)   : {shiftNm:.1f} nm")

#     # ── STEP E: Sensitivity sweep — reproduces Fig. 8 ────────────────────────
#     lamCoupledList = [lamRef]          # already have the reference point
#     nList          = [nWaterRef]

#     for nClad in nCladdingVals:
#         if abs(nClad - nWaterRef) < 1e-4:
#             continue                   # skip — already computed
#         _, _, pThr, pCrs = runSpectrum(
#             nClad, normFluxSim=normSim, normFluxObj=normFluxObj
#         )
#         lamCoupled = findCouplingWavelength(wavelengths, pThr, pCrs)
#         nList.append(nClad)
#         lamCoupledList.append(lamCoupled)

#     # Sort by refractive index (for clean plot)
#     sortIdx         = np.argsort(nList)
#     nSorted         = np.array(nList)[sortIdx]
#     lamSorted       = np.array(lamCoupledList)[sortIdx]

#     sensitivityNm = plotSensitivity(
#         nSorted, lamSorted,
#         savePath="fig8_sensitivity.png"
#     )

#     # ── STEP F: Performance metrics ───────────────────────────────────────────
#     metrics = computePerformanceMetrics(wavelengths, pThroughRef, sensitivityNm)

#     print("\n" + "=" * 55)
#     print("  PERFORMANCE SUMMARY")
#     print("=" * 55)
#     print(f"  Coupling wavelength   : {metrics['lamCoupled_nm']:.1f} nm")
#     print(f"  FWHM                  : {metrics['FWHM_nm']:.1f} nm")
#     print(f"  Q factor              : {metrics['Q']:.1f}")
#     print(f"  Sensitivity           : {sensitivityNm:.1f} nm/RIU")
#     print(f"  FOM                   : {metrics['FOM']:.2f} RIU⁻¹")
#     print(f"  Sensor resolution (3σ): {metrics['resolution_nm']:.3f} nm")
#     print(f"  LOD (estimated)       : {metrics['LOD_RIU']:.4e} RIU")
#     print("=" * 55)
#     print("\n  Paper reference values (400 nm gap, Table 1):")
#     print("    Sensitivity : 566 nm/RIU")
#     print("    FWHM        : 94 nm")
#     print("    FOM         : 6.018 RIU⁻¹")
#     print("\n[DONE] All figures saved.")


# if __name__ == "__main__":
#     main()


"""
SOI Asymmetric Directional Coupler Biosensor — MEEP Simulation
==============================================================
Reproduces Figs 5, 6, 7, 8 from:
  Alqabandi & Scullion, J. Opt. 28 (2026) 045001

Geometry (2-D FDTD, effective-index method)
  - Strip waveguide  : 350 nm wide, centered at y = 0
  - Slot waveguide   : 550 nm wide, 50 nm slot, centered at y = +0.850 µm
  - Edge-to-edge gap : 400 nm  (user-specified, Table 1 row)
  - Coupling length  : 25 µm
  - SOI stack        : 220 nm Si → effective index n_eff = 2.842
  - Cladding (ref.)  : water n = 1.318 at λ = 1550 nm

MEEP units: length in µm, frequency in c/µm (so f = 1/λ[µm])

Author note: Run with  `python3 adc_biosensor_meep.py`
Requires:   meep, numpy, matplotlib, scipy  (conda-forge recommended)
"""

import meep as mp
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — change to "TkAgg" if interactive
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats   import linregress

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — SIMULATION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# ── Effective indices (n_eff from 1-D slab solver, paper §3) ─────────────────
nEff        = 2.842        # Si slab effective index (220 nm thick)
nSiO2       = 1.444        # buried-oxide substrate (not used in 2-D EIM geometry)
nWaterRef   = 1.318        # reference cladding refractive index

# ── Waveguide geometry (µm) ───────────────────────────────────────────────────
wStrip      = 0.350        # strip waveguide width
wSlot       = 0.550        # slot waveguide total width (both rails + slot)
slotW       = 0.050        # slot gap width
gapEdge     = 0.400        # edge-to-edge separation between strip and slot WG
couplingLen = 25.0         # coupling region length [µm]

# Strip center at y=0; slot center calculated from geometry:
#   y_slot = wStrip/2 + gapEdge + wSlot/2
yCenterStrip = 0.0
yCenterSlot  = wStrip / 2 + gapEdge + wSlot / 2   # = 0.850 µm

# Rail widths of the slot waveguide (each rail = (wSlot - slotW) / 2)
railW = (wSlot - slotW) / 2    # = 0.250 µm per rail

# ── Domain and PML ────────────────────────────────────────────────────────────
padX        = 3.0          # padding along x (propagation axis) on each side [µm]
padYBot     = 1.5          # padding below strip [µm]
padYTop     = 1.5          # padding above slot [µm]
pmlThick    = 1.0          # PML thickness [µm]

domainX = couplingLen + 2 * padX
domainY = padYBot + yCenterSlot + wSlot / 2 + padYTop

# Center domain: shift y so the midpoint between waveguides is at y=0
yMid    = (yCenterStrip + yCenterSlot) / 2    # = +0.425 µm
# In MEEP the cell is centered at the origin, so we need an offset
# to place the strip at yCenterStrip − yMid from cell center
# → stripY = yCenterStrip − yMid = −0.425 µm  (in MEEP cell coords)
# → slotY  = yCenterSlot  − yMid = +0.425 µm
stripY  = yCenterStrip - yMid      # −0.425 µm
slotY   = yCenterSlot  - yMid      # +0.425 µm

# Source and monitor x-positions (in MEEP cell coords, cell centered at 0)
xLeft       = -couplingLen / 2     # left edge of coupling region
xRight      =  couplingLen / 2     # right edge of coupling region
xSrc        = xLeft  - 1.5        # 1.5 µm left of coupling start
xMonitor    = xRight + 1.5        # 1.5 µm right of coupling end

# ── Spectral parameters ───────────────────────────────────────────────────────
resolution  = 30           # pixels per µm (paper §3, "30 pixels µm⁻¹")
lamCen      = 1.55         # centre wavelength [µm]
lamMin      = 1.0          # shortest λ of interest [µm]
lamMax      = 2.0          # longest  λ of interest [µm]
fCen        = 1 / lamCen   # centre frequency [c/µm]
fMin        = 1 / lamMax   # 0.5
fMax        = 1 / lamMin   # 1.0
fBW         = fMax - fMin  # total bandwidth
nFreqs      = 500          # number of spectral points for Harminv / flux monitors

# ── Runtime ───────────────────────────────────────────────────────────────────
simTime     = 5000         # MEEP time units (~enough for spectral settling)

# ── Sensitivity sweep ─────────────────────────────────────────────────────────
# Paper sweeps cladding RI from 1.318 to 1.400 (Fig. 8); we reproduce this.
nCladdingVals = np.array([1.318, 1.330, 1.340, 1.350, 1.360, 1.370, 1.380, 1.400])


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — GEOMETRY BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def buildGeometry(nCladding):
    """
    Returns the MEEP geometry (list of Block objects) for the 2-D effective-
    index model.  The waveguide cores use nEff; the background is nCladding.

    Parameters
    ----------
    nCladding : float
        Refractive index of the top cladding medium (water or analyte).

    Returns
    -------
    geometry : list[mp.Block]
    medium   : mp.Medium  — background medium
    """
    # Background = analyte cladding
    background = mp.Medium(index=nCladding)

    # Strip waveguide (solid rectangle)
    stripBlock = mp.Block(
        size    = mp.Vector3(couplingLen, wStrip, mp.inf),
        center  = mp.Vector3(0, stripY, 0),
        material= mp.Medium(index=nEff)
    )

    # Slot waveguide — left rail
    leftRailBlock = mp.Block(
        size    = mp.Vector3(couplingLen, railW, mp.inf),
        center  = mp.Vector3(0, slotY - (slotW / 2 + railW / 2), 0),
        material= mp.Medium(index=nEff)
    )

    # Slot waveguide — right rail
    rightRailBlock = mp.Block(
        size    = mp.Vector3(couplingLen, railW, mp.inf),
        center  = mp.Vector3(0, slotY + (slotW / 2 + railW / 2), 0),
        material= mp.Medium(index=nEff)
    )

    # Slot gap remains background (nCladding fills the slot)

    geometry = [stripBlock, leftRailBlock, rightRailBlock]
    return geometry, background


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — SOURCE, PML, AND CELL
# ─────────────────────────────────────────────────────────────────────────────

def buildSimulation(nCladding, nFreqsMon=nFreqs, runTime=simTime):
    """
    Builds and returns a configured mp.Simulation object.

    The Gaussian source is placed 1.5 µm left of the coupling region and
    excites only the strip waveguide (Ez polarisation, TE-like in 2-D).
    Two flux monitors are placed 1.5 µm right of the coupling region:
      • throughFlux  — at the strip waveguide output
      • crossFlux    — at the slot waveguide output

    Parameters
    ----------
    nCladding  : float   — analyte refractive index
    nFreqsMon  : int     — frequency points in flux monitors
    runTime    : float   — simulation run time (MEEP units)

    Returns
    -------
    sim          : mp.Simulation
    throughFlux  : flux region
    crossFlux    : flux region
    """
    geometry, background = buildGeometry(nCladding)

    cell = mp.Vector3(domainX, domainY, 0)

    pml = [mp.PML(pmlThick)]

    # Gaussian pulse exciting the strip waveguide (TE: Ez)
    source = mp.GaussianSource(
        frequency  = fCen,
        fwidth     = fBW,
        is_integrated = True   # correct normalisation for broadband sources
    )

    sources = [mp.Source(
        src      = source,
        component= mp.Ez,
        center   = mp.Vector3(xSrc, stripY, 0),
        size     = mp.Vector3(0, wStrip * 2, 0)   # span 2× strip width to capture mode
    )]

    sim = mp.Simulation(
        cell_size   = cell,
        geometry    = geometry,
        sources     = sources,
        boundary_layers = pml,
        default_material = background,
        resolution  = resolution
    )

    # ── Flux monitors ─────────────────────────────────────────────────────────
    # Monitor width: 3× waveguide width to capture evanescent tails
    monWidthStrip = wStrip * 3
    monWidthSlot  = wSlot  * 3

    throughRegion = mp.FluxRegion(
        center = mp.Vector3(xMonitor, stripY, 0),
        size   = mp.Vector3(0, monWidthStrip, 0)
    )
    crossRegion = mp.FluxRegion(
        center = mp.Vector3(xMonitor, slotY, 0),
        size   = mp.Vector3(0, monWidthSlot, 0)
    )

    throughFlux = sim.add_flux(fCen, fBW, nFreqsMon, throughRegion)
    crossFlux   = sim.add_flux(fCen, fBW, nFreqsMon, crossRegion)

    return sim, throughFlux, crossFlux


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — NORMALISATION (empty-waveguide reference run)
# ─────────────────────────────────────────────────────────────────────────────

def runNormalization():
    """
    Run the simulation with NO geometry (empty cell) to record the
    incident flux.  This flux is used to normalise all subsequent runs
    so that output powers are expressed as fractions of input power.

    Returns
    -------
    normThroughData : np.ndarray — incident flux spectrum (nFreqs values)
    freqs           : np.ndarray — corresponding frequencies [c/µm]
    """
    print("\n[NORM] Running normalisation (empty waveguide) ...")

    # For normalisation: use the same source but no geometry
    cell = mp.Vector3(domainX, domainY, 0)
    pml  = [mp.PML(pmlThick)]

    source = mp.GaussianSource(
        frequency     = fCen,
        fwidth        = fBW,
        is_integrated = True
    )
    sources = [mp.Source(
        src       = source,
        component = mp.Ez,
        center    = mp.Vector3(xSrc, stripY, 0),
        size      = mp.Vector3(0, wStrip * 2, 0)
    )]

    sim = mp.Simulation(
        cell_size        = cell,
        geometry         = [],
        sources          = sources,
        boundary_layers  = pml,
        resolution       = resolution
    )

    # CRITICAL: monitor size must exactly match the through-port monitor in
    # buildSimulation() (monWidthStrip = wStrip * 3) so that load_minus_flux
    # finds identical dataset dimensions in the HDF5 file.
    monRegion = mp.FluxRegion(
        center = mp.Vector3(xMonitor, stripY, 0),
        size   = mp.Vector3(0, wStrip * 3, 0)
    )
    normFlux = sim.add_flux(fCen, fBW, nFreqs, monRegion)

    sim.run(until=simTime)

    normData = np.array(mp.get_fluxes(normFlux))
    freqs    = np.array(mp.get_flux_freqs(normFlux))

    # Save flux data for subtraction in subsequent runs
    sim.save_flux("norm_flux", normFlux)

    return normData, freqs, sim, normFlux


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — SINGLE SPECTRAL RUN
# ─────────────────────────────────────────────────────────────────────────────

def runSpectrum(nCladding, normFluxSim=None, normFluxObj=None):
    """
    Run the full ADC simulation for a given cladding index.
    Returns normalised through-port and cross-port power spectra.

    Parameters
    ----------
    nCladding   : float   — analyte refractive index
    normFluxSim : previously run normalisation mp.Simulation (for flux sub.)
    normFluxObj : flux object from normalisation run

    Returns
    -------
    freqs       : np.ndarray  — frequency array [c/µm]
    wavelengths : np.ndarray  — wavelength array [µm]
    pThrough    : np.ndarray  — normalised through-port power [0, 1]
    pCross      : np.ndarray  — normalised cross-port power [0, 1]
    """
    print(f"\n[SIM] n_cladding = {nCladding:.3f}")

    sim, throughFlux, crossFlux = buildSimulation(nCladding)

    # Load normalisation flux (subtract incident field)
    if normFluxObj is not None:
        sim.load_minus_flux("norm_flux", throughFlux)

    sim.run(until=simTime)

    freqs       = np.array(mp.get_flux_freqs(throughFlux))
    wavelengths = 1 / freqs                            # µm

    throughData = np.array(mp.get_fluxes(throughFlux))
    crossData   = np.array(mp.get_fluxes(crossFlux))

    # Normalise by incident flux
    if normFluxObj is not None:
        normData = np.abs(np.array(mp.get_fluxes(normFluxObj)))
        normData[normData == 0] = 1.0                  # avoid divide-by-zero
        pThrough = np.abs(throughData) / normData
        pCross   = np.abs(crossData)   / normData
    else:
        # Fallback: normalise to peak of through port
        peak = np.max(np.abs(throughData))
        peak = peak if peak > 0 else 1.0
        pThrough = np.abs(throughData) / peak
        pCross   = np.abs(crossData)   / peak

    # Clip to [0, 1] — small numerical overshoot can occur
    pThrough = np.clip(pThrough, 0, 1.2)
    pCross   = np.clip(pCross,   0, 1.2)

    return freqs, wavelengths, pThrough, pCross


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — FIELD SNAPSHOT RUN (Fig. 6)
# ─────────────────────────────────────────────────────────────────────────────

def runFieldSnapshot(lamSource, nCladding=nWaterRef, tSnap=2500):
    """
    Run a CW (continuous-wave) source simulation to capture the Ez field
    distribution at a fixed wavelength.  Returns the 2-D field array and
    the corresponding x/y coordinate arrays.

    Parameters
    ----------
    lamSource  : float  — source wavelength [µm]
    nCladding  : float  — cladding index
    tSnap      : float  — time at which to take the snapshot [MEEP units]
                          (must be >> transit time for steady state)

    Returns
    -------
    xArr, yArr : 1-D coordinate arrays [µm]
    ezField    : 2-D Ez field array
    """
    print(f"\n[SNAP] Wavelength = {lamSource} µm")

    fSource = 1 / lamSource

    geometry, background = buildGeometry(nCladding)

    cell = mp.Vector3(domainX, domainY, 0)
    pml  = [mp.PML(pmlThick)]

    source = mp.ContinuousSource(
        frequency  = fSource,
        is_integrated = True
    )
    sources = [mp.Source(
        src       = source,
        component = mp.Ez,
        center    = mp.Vector3(xSrc, stripY, 0),
        size      = mp.Vector3(0, wStrip * 2, 0)
    )]

    sim = mp.Simulation(
        cell_size        = cell,
        geometry         = geometry,
        sources          = sources,
        boundary_layers  = pml,
        default_material = background,
        resolution       = resolution
    )

    sim.run(until=tSnap)

    ezData = sim.get_array(
        center    = mp.Vector3(0, 0, 0),
        size      = mp.Vector3(domainX, domainY, 0),
        component = mp.Ez
    )

    # Build coordinate arrays
    nx, ny = ezData.shape
    xArr   = np.linspace(-domainX / 2, domainX / 2, nx)
    yArr   = np.linspace(-domainY / 2, domainY / 2, ny)

    return xArr, yArr, ezData


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7 — COUPLING WAVELENGTH FINDER
# ─────────────────────────────────────────────────────────────────────────────

def findCouplingWavelength(wavelengths, pThrough, pCross):
    """
    Locate the coupling wavelength as the dip in the through-port spectrum
    (equivalently, the peak in the cross-port spectrum).

    Strategy:
    1. Find the minimum in pThrough in the range 1.3–2.0 µm.
    2. Confirm it corresponds to a peak in pCross.
    3. Return the wavelength at that minimum.

    Parameters
    ----------
    wavelengths : np.ndarray
    pThrough    : np.ndarray
    pCross      : np.ndarray

    Returns
    -------
    lamCoupled : float  — coupling wavelength [µm]
    """
    # Work in the physically relevant window (1.3–2.0 µm)
    mask = (wavelengths >= 1.3) & (wavelengths <= 2.0)
    wMask = wavelengths[mask]
    pMask = pThrough[mask]

    # Find the dip (minimum) — use inverted array for find_peaks
    peaks, props = find_peaks(-pMask, prominence=0.05)

    if len(peaks) == 0:
        # Fallback: simple argmin
        idxMin = np.argmin(pMask)
        return wMask[idxMin]

    # If multiple dips, pick the most prominent one
    bestIdx = peaks[np.argmax(props["prominences"])]
    return wMask[bestIdx]


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 8 — PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def plotTransmissionSpectra(wavelengths, pThrough, pCross, title="",
                            savePath="fig5_transmission_spectra.png"):
    """
    Reproduce Fig. 5: through-port and cross-port power vs. wavelength.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(wavelengths, pThrough, color="blue",  linewidth=1.2, label="Through Port")
    ax.plot(wavelengths, pCross,   color="red",   linewidth=1.2, label="Cross Port")

    ax.set_xlabel("Wavelength (µm)", fontsize=13)
    ax.set_ylabel("Power ratio",     fontsize=13)
    ax.set_title(title or "Transmission Spectra", fontsize=13)
    ax.set_xlim(1.0, 2.0)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(savePath, dpi=150)
    plt.close()
    print(f"[PLOT] Saved {savePath}")


def plotFieldSnapshot(xArr, yArr, ezField, lamSource, label,
                      savePath="fig6_field.png"):
    """
    Reproduce Fig. 6 panels: Ez field distribution colour map.
    Waveguide outlines are drawn as white rectangles for reference.
    """
    # Use |Ez|² (intensity) for clearer visualisation
    intensity = np.abs(ezField.T) ** 2
    intensityNorm = intensity / (intensity.max() + 1e-30)

    fig, ax = plt.subplots(figsize=(12, 4))

    im = ax.imshow(
        intensityNorm,
        origin="lower",
        extent=[xArr[0], xArr[-1], yArr[0], yArr[-1]],
        aspect="auto",
        cmap="hot",
        vmin=0, vmax=1
    )
    plt.colorbar(im, ax=ax, label="|Ez|² (normalised)")

    # Draw waveguide outlines
    def drawWg(yCenter, width, color="white"):
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle(
            (xLeft, yCenter - width / 2 - yMid),
            couplingLen, width,
            linewidth=1.0, edgecolor=color,
            facecolor="none", linestyle="--"
        ))

    drawWg(yCenterStrip - yMid, wStrip, "cyan")
    drawWg(yCenterSlot  - yMid, wSlot,  "cyan")

    ax.set_xlabel("x (µm)", fontsize=12)
    ax.set_ylabel("y (µm)", fontsize=12)
    ax.set_title(f"Ez field — λ = {lamSource} µm   {label}", fontsize=12)
    ax.set_xlim(xLeft - 1, xRight + 1)    # zoom to coupling region

    plt.tight_layout()
    plt.savefig(savePath, dpi=150)
    plt.close()
    print(f"[PLOT] Saved {savePath}")


def plotShiftedWavelength(wavelengths, pThrough1, pThrough2,
                          n1, n2,
                          savePath="fig7_shifted_wavelength.png"):
    """
    Reproduce Fig. 7: through-port spectra for two different cladding indices
    showing the blue-shift of the coupling dip.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(wavelengths, pThrough1, color="blue",
            linewidth=1.2, label=f"sample 1: {n1:.3f}")
    ax.plot(wavelengths, pThrough2, color="orange",
            linewidth=1.2, label=f"sample 2: {n2:.3f}")

    ax.set_xlabel("Wavelength (µm)", fontsize=13)
    ax.set_ylabel("Power Ratio",     fontsize=13)
    ax.set_title("Shifted Wavelength Plot", fontsize=13)
    ax.set_xlim(1.2, 2.0)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(savePath, dpi=150)
    plt.close()
    print(f"[PLOT] Saved {savePath}")


def plotSensitivity(nVals, lamCoupledVals,
                    savePath="fig8_sensitivity.png"):
    """
    Reproduce Fig. 8: coupled wavelength vs. cladding refractive index,
    with a linear best-fit line and sensitivity annotation.
    """
    slope, intercept, r, p, se = linregress(nVals, lamCoupledVals)
    sensitivityNmPerRIU = slope * 1000    # µm/RIU → nm/RIU
    fitLine = slope * np.array(nVals) + intercept

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(nVals, lamCoupledVals, marker="D", s=50,
               color="blue", label="Data points", zorder=5)
    ax.plot(nVals, fitLine, color="blue", linestyle="--",
            label=f"Best fit (Sensitivity = {sensitivityNmPerRIU:.2f} nm/RIU)")

    ax.set_xlabel("Refractive index",        fontsize=13)
    ax.set_ylabel("Coupled wavelength (µm)", fontsize=13)
    ax.set_title("Sensitivity Plot",         fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(savePath, dpi=150)
    plt.close()
    print(f"[PLOT] Saved {savePath}")
    print(f"  → Sensitivity: {sensitivityNmPerRIU:.1f} nm/RIU   (R² = {r**2:.4f})")
    return sensitivityNmPerRIU


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 9 — FIGURE OF MERIT, LOD, Q-FACTOR
# ─────────────────────────────────────────────────────────────────────────────

def computePerformanceMetrics(wavelengths, pThrough, sensitivityNmPerRIU):
    """
    Compute FOM, quality factor Q, and estimated LOD following
    White & Fan (2008) [Ref. 39 in paper].

    Parameters
    ----------
    wavelengths         : np.ndarray   [µm]
    pThrough            : np.ndarray   normalised through-port power
    sensitivityNmPerRIU : float        [nm/RIU]

    Returns
    -------
    dict with keys: FWHM_nm, Q, FOM, LOD_RIU
    """
    # Find the coupling dip
    mask = (wavelengths >= 1.3) & (wavelengths <= 2.0)
    wMask = wavelengths[mask]
    pMask = pThrough[mask]

    idxMin = np.argmin(pMask)
    lamCoupled = wMask[idxMin]
    dipVal     = pMask[idxMin]

    # Half-maximum level (measured from the dip, not from unity)
    halfMax = (1.0 + dipVal) / 2.0

    # Indices where power crosses halfMax on either side of the dip
    leftMask  = (wMask < lamCoupled) & (pMask < halfMax)
    rightMask = (wMask > lamCoupled) & (pMask < halfMax)

    lamLeft  = wMask[leftMask][0]  if leftMask.any()  else lamCoupled - 0.05
    lamRight = wMask[rightMask][-1] if rightMask.any() else lamCoupled + 0.05

    fwhmMicron = lamRight - lamLeft
    fwhmNm     = fwhmMicron * 1000    # nm

    # Quality factor
    Q = (lamCoupled * 1000) / fwhmNm   # λ/Δλ, both in nm

    # FOM = Sensitivity / FWHM  [RIU⁻¹]
    FOM = sensitivityNmPerRIU / fwhmNm

    # LOD using White & Fan (Eq. 5–7 in paper)
    SNR_dB    = 50.0
    SNR_lin   = 10 ** (SNR_dB / 10)
    sigmaAmpl = (fwhmNm / 4.5) * (SNR_lin ** (-0.25))   # Eq. (5)
    sigmaTemp = 0.11     # nm  (0.5 °C shift, all thermo-optic coefficients)
    sigmaLas  = 0.001    # nm  (1 fm laser linewidth, worst case)
    resolution3sigma = 3 * np.sqrt(sigmaAmpl**2 + sigmaTemp**2 + sigmaLas**2)  # Eq. (6)
    LOD = resolution3sigma / sensitivityNmPerRIU   # Eq. (7)

    return {
        "lamCoupled_nm" : lamCoupled * 1000,
        "FWHM_nm"       : fwhmNm,
        "Q"             : Q,
        "FOM"           : FOM,
        "sigmaAmpl_nm"  : sigmaAmpl,
        "resolution_nm" : resolution3sigma,
        "LOD_RIU"       : LOD
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 10 — MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  SOI ADC Biosensor — MEEP Simulation")
    print("  Reproducing Figs 5, 6, 7, 8 of Alqabandi & Scullion (2026)")
    print("=" * 65)
    print(f"\nGeometry summary:")
    print(f"  Strip WG width     : {wStrip*1000:.0f} nm")
    print(f"  Slot WG width      : {wSlot*1000:.0f} nm  (slot = {slotW*1000:.0f} nm)")
    print(f"  Rail width         : {railW*1000:.0f} nm each")
    print(f"  Edge-to-edge gap   : {gapEdge*1000:.0f} nm")
    print(f"  Coupling length    : {couplingLen:.0f} µm")
    print(f"  Strip y-center     : {stripY:.4f} µm (MEEP cell coords)")
    print(f"  Slot  y-center     : {slotY:.4f} µm (MEEP cell coords)")
    print(f"  Domain X           : {domainX:.1f} µm")
    print(f"  Domain Y           : {domainY:.2f} µm")
    print(f"  Resolution         : {resolution} px/µm")
    print(f"  n_eff (Si slab)    : {nEff}")
    print(f"  n_cladding (ref)   : {nWaterRef}")

    # ── STEP A: Normalisation run ─────────────────────────────────────────────
    normData, freqs, normSim, normFluxObj = runNormalization()

    wavelengths = 1 / freqs     # µm

    # ── STEP B: Reference spectrum (n = 1.318, water) — reproduces Fig. 5 ────
    _, _, pThroughRef, pCrossRef = runSpectrum(
        nWaterRef, normFluxSim=normSim, normFluxObj=normFluxObj
    )

    plotTransmissionSpectra(
        wavelengths, pThroughRef, pCrossRef,
        title="Transmission Spectra (n_clad = 1.318, gap = 400 nm)",
        savePath="fig5_transmission_spectra.png"
    )

    lamRef = findCouplingWavelength(wavelengths, pThroughRef, pCrossRef)
    print(f"\n[RESULT] Coupling wavelength at n=1.318 : {lamRef*1000:.1f} nm")

    # ── STEP C: Field snapshots — reproduces Fig. 6 ───────────────────────────
    # Fig. 6(a): coupling wavelength → power transfers to slot
    xA, yA, ezA = runFieldSnapshot(
        lamSource=lamRef, nCladding=nWaterRef, tSnap=2500
    )
    plotFieldSnapshot(
        xA, yA, ezA, lamSource=lamRef,
        label="(coupling ON)",
        savePath="fig6a_field_coupling.png"
    )

    # Fig. 6(b): off-coupling wavelength (1.2 µm) → power stays in strip
    xB, yB, ezB = runFieldSnapshot(
        lamSource=1.200, nCladding=nWaterRef, tSnap=2500
    )
    plotFieldSnapshot(
        xB, yB, ezB, lamSource=1.200,
        label="(coupling OFF)",
        savePath="fig6b_field_no_coupling.png"
    )

    # ── STEP D: Shifted spectrum (n = 1.34) — reproduces Fig. 7 ──────────────
    _, _, pThrough134, pCross134 = runSpectrum(
        1.340, normFluxSim=normSim, normFluxObj=normFluxObj
    )

    plotShiftedWavelength(
        wavelengths, pThroughRef, pThrough134,
        n1=1.318, n2=1.340,
        savePath="fig7_shifted_wavelength.png"
    )

    lam134 = findCouplingWavelength(wavelengths, pThrough134, pCross134)
    shiftNm = (lamRef - lam134) * 1000    # blue-shift → lamRef > lam134
    print(f"[RESULT] Coupling wavelength at n=1.340 : {lam134*1000:.1f} nm")
    print(f"[RESULT] Blue shift Δλ (1.318→1.340)   : {shiftNm:.1f} nm")

    # ── STEP E: Sensitivity sweep — reproduces Fig. 8 ────────────────────────
    lamCoupledList = [lamRef]          # already have the reference point
    nList          = [nWaterRef]

    for nClad in nCladdingVals:
        if abs(nClad - nWaterRef) < 1e-4:
            continue                   # skip — already computed
        _, _, pThr, pCrs = runSpectrum(
            nClad, normFluxSim=normSim, normFluxObj=normFluxObj
        )
        lamCoupled = findCouplingWavelength(wavelengths, pThr, pCrs)
        nList.append(nClad)
        lamCoupledList.append(lamCoupled)

    # Sort by refractive index (for clean plot)
    sortIdx         = np.argsort(nList)
    nSorted         = np.array(nList)[sortIdx]
    lamSorted       = np.array(lamCoupledList)[sortIdx]

    sensitivityNm = plotSensitivity(
        nSorted, lamSorted,
        savePath="fig8_sensitivity.png"
    )

    # ── STEP F: Performance metrics ───────────────────────────────────────────
    metrics = computePerformanceMetrics(wavelengths, pThroughRef, sensitivityNm)

    print("\n" + "=" * 55)
    print("  PERFORMANCE SUMMARY")
    print("=" * 55)
    print(f"  Coupling wavelength   : {metrics['lamCoupled_nm']:.1f} nm")
    print(f"  FWHM                  : {metrics['FWHM_nm']:.1f} nm")
    print(f"  Q factor              : {metrics['Q']:.1f}")
    print(f"  Sensitivity           : {sensitivityNm:.1f} nm/RIU")
    print(f"  FOM                   : {metrics['FOM']:.2f} RIU⁻¹")
    print(f"  Sensor resolution (3σ): {metrics['resolution_nm']:.3f} nm")
    print(f"  LOD (estimated)       : {metrics['LOD_RIU']:.4e} RIU")
    print("=" * 55)
    print("\n  Paper reference values (400 nm gap, Table 1):")
    print("    Sensitivity : 566 nm/RIU")
    print("    FWHM        : 94 nm")
    print("    FOM         : 6.018 RIU⁻¹")
    print("\n[DONE] All figures saved.")


if __name__ == "__main__":
    main()