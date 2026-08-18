"""
fig7fig8_sensitivity.py
=========================
Reproduces figures 7 and 8: transmission-level shift for two analyte
refractive indices (7), and the coupled-wavelength vs refractive-index
sensitivity fit across the paper's full RI range 1.318-1.400 (8).

Uses the coupling length found by fig5_transmissionSpectra.py's sweep -
set COUPLING_LENGTH_UM below to that value before running (25 um is only
the starting guess, see the notes in the chat reply).

Run:
    python fig7fig8_sensitivity.py
"""

import numpy as np
import matplotlib.pyplot as plt

import commonConfig as cfg
from fig5_transmissionSpectra import normalisedSpectra

# Set this to the length fig5_transmissionSpectra.py's sweep actually
# selected (deepest through-port dip) before running this script.
COUPLING_LENGTH_UM = cfg.couplingLengthGuess


def findCoupledWavelength(wavelengthsNm, throughRatio,
                           searchRangeNm=(1300, 1800)):
    """Coupled/dip wavelength = minimum of the through-port spectrum,
    restricted to a sensible search window away from the sweep edges."""
    mask = (wavelengthsNm >= searchRangeNm[0]) & (wavelengthsNm <= searchRangeNm[1])
    idx = np.argmin(throughRatio[mask])
    return wavelengthsNm[mask][idx]


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # Figure 7: through-port spectrum shift between two analyte samples
    # -----------------------------------------------------------------
    sample1Ri, sample2Ri = 1.318, 1.34
    wl1, through1, _ = normalisedSpectra(COUPLING_LENGTH_UM, cfg.gapWidth, sample1Ri)
    wl2, through2, _ = normalisedSpectra(COUPLING_LENGTH_UM, cfg.gapWidth, sample2Ri)

    dip1 = findCoupledWavelength(wl1, through1)
    dip2 = findCoupledWavelength(wl2, through2)
    print(f"Sample 1 (RI={sample1Ri}): coupled wavelength = {dip1:.1f} nm")
    print(f"Sample 2 (RI={sample2Ri}): coupled wavelength = {dip2:.1f} nm")
    print(f"Blue shift = {dip1 - dip2:.1f} nm")

    plt.figure(figsize=(7, 5))
    plt.plot(wl1 / 1000, through1, label=f"sample 1: {sample1Ri}")
    plt.plot(wl2 / 1000, through2, label=f"sample 2: {sample2Ri}")
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Power Ratio")
    plt.title("Shifted Wavelength Plot (through port)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig7_shiftedWavelength.png", dpi=200)
    print("Saved fig7_shiftedWavelength.png")

    # -----------------------------------------------------------------
    # Figure 8: coupled wavelength vs refractive index, linear fit
    # -----------------------------------------------------------------
    coupledWavelengths = []
    for ri in cfg.sensitivityRiList:
        wl, through, _ = normalisedSpectra(COUPLING_LENGTH_UM, cfg.gapWidth, ri)
        coupledWavelengths.append(findCoupledWavelength(wl, through))
        print(f"  RI = {ri:.3f}  ->  coupled wavelength = {coupledWavelengths[-1]:.1f} nm")

    riArray = np.array(cfg.sensitivityRiList)
    wavelengthArray = np.array(coupledWavelengths)
    slope, intercept = np.polyfit(riArray, wavelengthArray, 1)
    print(f"\nBulk sensitivity (linear fit slope) = {slope:.1f} nm/RIU")

    plt.figure(figsize=(7, 5))
    plt.plot(riArray, wavelengthArray / 1000, 'o', label="Data points")
    plt.plot(riArray, (slope * riArray + intercept) / 1000, '--',
              label=f"Best fit (Sensitivity = {slope:.2f} nm/RIU)")
    plt.xlabel("Refractive index")
    plt.ylabel("Coupled wavelength (um)")
    plt.title("Sensitivity Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig8_sensitivityPlot.png", dpi=200)
    print("Saved fig8_sensitivityPlot.png")
