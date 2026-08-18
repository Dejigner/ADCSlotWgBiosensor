"""
geometryBuilder.py
===================
Builds the 2D (top-down, effective-index) MEEP geometry for the straight +
slot asymmetric directional coupler. Propagation is along x, the two
waveguides are separated along y. The vertical (z) confinement of the 220 nm
silicon slab is folded into the silicon effective index (2.842), exactly as
described in the paper's Section 3, so no z-dimension is simulated.
"""

import meep as mp
import commonConfig as cfg


def transverseLayout(gapWidth=cfg.gapWidth):
    """
    Returns the y-center positions of the three silicon features
    (straight arm, slot left rail, slot right rail) for a given
    edge-to-edge gap width, plus the total transverse span needed.

    The straight and slot arms are generally different widths, so the raw
    layout is NOT symmetric about y = 0. To keep equal cladding padding
    (and hence equal PML separation) on both sides of the simulation cell,
    the whole layout is re-centred here so its outer edges sit at +/- half
    of the total silicon span.
    """
    rawStraightCenterY = -(gapWidth / 2 + cfg.straightWaveguideWidth / 2)
    rawLeftRailCenterY = gapWidth / 2 + cfg.slotRailWidth / 2
    rawRightRailCenterY = (gapWidth / 2 + cfg.slotRailWidth
                            + cfg.slotGapWidth + cfg.slotRailWidth / 2)

    yBottomEdge = rawStraightCenterY - cfg.straightWaveguideWidth / 2
    yTopEdge = rawRightRailCenterY + cfg.slotRailWidth / 2
    totalSiliconSpan = yTopEdge - yBottomEdge

    recenterOffset = (yBottomEdge + yTopEdge) / 2  # midpoint of the raw layout
    straightCenterY = rawStraightCenterY - recenterOffset
    slotLeftRailCenterY = rawLeftRailCenterY - recenterOffset
    slotRightRailCenterY = rawRightRailCenterY - recenterOffset

    return straightCenterY, slotLeftRailCenterY, slotRightRailCenterY, totalSiliconSpan


def buildCellAndGeometry(deviceLength, gapWidth=cfg.gapWidth, includeSlotArm=True):
    """
    Build the simulation cell size and the list of geometric silicon blocks.

    deviceLength : straight parallel run length (um) - this is the quantity
                   quoted as "coupling length" in the paper's table 1 /
                   figure 7 caption.
    includeSlotArm : set False to build a bare straight-waveguide-only cell,
                     used for the incident-power normalization run.
    """
    strY, leftRailY, rightRailY, siliconSpan = transverseLayout(gapWidth)

    yPadding = 0.8  # um, cladding padding so evanescent tails decay before PML
    cellSizeY = siliconSpan + 2 * yPadding + 2 * cfg.pmlThickness
    cellSizeX = deviceLength + 2 * cfg.leadLength + 2 * cfg.pmlThickness

    cellSize = mp.Vector3(cellSizeX, cellSizeY, 0)

    geometry = [
        mp.Block(size=mp.Vector3(mp.inf, cfg.straightWaveguideWidth, mp.inf),
                 center=mp.Vector3(0, strY, 0),
                 material=cfg.siliconMedium),
    ]

    if includeSlotArm:
        geometry += [
            mp.Block(size=mp.Vector3(mp.inf, cfg.slotRailWidth, mp.inf),
                     center=mp.Vector3(0, leftRailY, 0),
                     material=cfg.siliconMedium),
            mp.Block(size=mp.Vector3(mp.inf, cfg.slotRailWidth, mp.inf),
                     center=mp.Vector3(0, rightRailY, 0),
                     material=cfg.siliconMedium),
        ]

    return cellSize, geometry, strY, leftRailY, rightRailY
