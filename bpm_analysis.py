#!/usr/bin/env python3
from __future__ import annotations
"""
BAR BPM analysis tool.

Purpose
-------
This script reproduces the main electrostatic BPM studies discussed in
`README.md`:

1. longitundinal button signal formation and filtering
2. 2D boundary-element BPM position reconstruction
3. linearity, polynomial correction, and resolution plots
4. BAR-style signal-chain figures such as Fig. 3/4/5/7/8/9

How to use
----------
Run the script with one YAML input file:

    python3 bpm_analysis.py /absolute/path/to/config.yaml

Expected outputs
----------------
The script writes figures and a Markdown report into the output directory defined
by the YAML file. Typical outputs include:

- figure_3_impedance_and_image_current_fft.png
- figure_4_image_current_and_button_voltage.png
- figure_5_cable_input_output.png
- figure_7_signal_fft_and_filter_response.png
- figure_8_filter_output_voltage.png
- figure_9_button_voltage_cases.png
- figure_11_linearity.png
- figure_12_polyfit.png
- figure_13_resolution.png
- analysis_report.md

Version metadata
----------------
- Initialized date: 2026-05-05
- Last updated: 2026-05-18
- Version: 0.6.1

Maintenance note
----------------
Keep the physics comments in this file aligned with the equations documented in
`README.md`, especially the labeled equations in Section 1.
"""

import argparse
import copy
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

# Matplotlib cannot write to the default cache directory in this environment.
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-bpm")

import matplotlib
import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.path import Path as MplPath
from numpy.polynomial.legendre import leggauss
from scipy import constants
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.signal import bessel, butter, sosfilt, sosfreqz

matplotlib.use("Agg")


@dataclass
class Boundary:
    """Discrete 2D chamber boundary used by the BEM solve.

    Attributes:
        points:
            Boundary vertices or sampled points in mm, ordered around the chamber.
        seg_start, seg_end:
            Start and end points of each boundary segment in mm.
        midpoints:
            Segment midpoints in mm. These act as collocation points in the BEM.
        tangents:
            Unit tangent vector for each segment.
        lengths:
            Segment lengths in mm.
        perimeter:
            Total chamber perimeter in mm.
        kind:
            Geometry type such as ``round``, ``ellipse``, or ``polygon``.
    """
    points: np.ndarray
    seg_start: np.ndarray
    seg_end: np.ndarray
    midpoints: np.ndarray
    tangents: np.ndarray
    lengths: np.ndarray
    perimeter: float
    kind: str


def load_config(path: Path) -> dict[str, Any]:
    """Load one YAML configuration file.

    Args:
        path:
            Absolute or relative path to the YAML input file.

    Returns:
        Parsed YAML content as a nested Python dictionary.
    """
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def mm_to_m(values: Any) -> Any:
    """Convert millimeter-valued scalars or arrays to meters.

    Args:
        values:
            Scalar, list, or NumPy-compatible array in mm.

    Returns:
        NumPy array or scalar-like object converted to meters.
    """
    return np.asarray(values, dtype=float) * 1e-3


def ensure_dir(path: Path) -> None:
    """Create an output directory if it does not already exist.

    Args:
        path:
            Directory path to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def close_boundary(points: np.ndarray) -> np.ndarray:
    """Ensure that a boundary point list is explicitly closed.

    Args:
        points:
            Array of shape ``(N, 2)`` describing ordered boundary points.

    Returns:
        Same point list if already closed, otherwise a new array with the first
        point appended at the end.
    """
    if np.allclose(points[0], points[-1]):
        return points.copy()
    return np.vstack([points, points[0]])


def sample_polygon(points_mm: list[list[float]], n_segments: int) -> np.ndarray:
    """Resample a polygon boundary into approximately uniform segments.

    Args:
        points_mm:
            Polygon corner points in mm, ordered around the chamber.
        n_segments:
            Target total number of boundary segments.

    Returns:
        Array of resampled polygon points in mm.
    """
    points = np.asarray(points_mm, dtype=float)
    closed = close_boundary(points)
    edge_vec = np.diff(closed, axis=0)
    edge_len = np.linalg.norm(edge_vec, axis=1)
    total = edge_len.sum()
    counts = np.maximum(1, np.round(n_segments * edge_len / total).astype(int))
    sampled: list[np.ndarray] = []
    for start, end, count in zip(closed[:-1], closed[1:], counts):
        t = np.linspace(0.0, 1.0, count, endpoint=False)
        sampled.append(start[None, :] + (end - start)[None, :] * t[:, None])
    return np.vstack(sampled)


def build_boundary(chamber_cfg: dict[str, Any]) -> Boundary:
    """Build the discrete chamber boundary from YAML geometry settings.

    Supported geometries are described in `README.md` Section 2.3.

    Args:
        chamber_cfg:
            The ``chamber`` block from the YAML file.

    Returns:
        A :class:`Boundary` object containing the fully discretized cross-section.
    """
    kind = chamber_cfg["kind"].lower()
    n_segments = int(chamber_cfg.get("boundary_elements", 320))

    if kind == "round":
        radius = float(chamber_cfg["radius_mm"])
        theta = np.linspace(0.0, 2.0 * math.pi, n_segments, endpoint=False)
        points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    elif kind == "ellipse":
        a = float(chamber_cfg["a_mm"])
        b = float(chamber_cfg["b_mm"])
        theta = np.linspace(0.0, 2.0 * math.pi, n_segments, endpoint=False)
        points = np.column_stack([a * np.cos(theta), b * np.sin(theta)])
    elif kind == "polygon":
        points = sample_polygon(chamber_cfg["points_mm"], n_segments)
    else:
        raise ValueError(f"Unsupported chamber kind: {kind}")

    seg_start = points
    seg_end = np.roll(points, -1, axis=0)
    seg_vec = seg_end - seg_start
    lengths = np.linalg.norm(seg_vec, axis=1)
    tangents = seg_vec / lengths[:, None]
    midpoints = 0.5 * (seg_start + seg_end)
    perimeter = float(lengths.sum())
    return Boundary(
        points=points,
        seg_start=seg_start,
        seg_end=seg_end,
        midpoints=midpoints,
        tangents=tangents,
        lengths=lengths,
        perimeter=perimeter,
        kind=kind,
    )


def inside_chamber(boundary: Boundary, chamber_cfg: dict[str, Any], xy_mm: np.ndarray) -> np.ndarray:
    """Test whether transverse sample points lie inside the chamber aperture.

    Args:
        boundary:
            Discrete chamber boundary.
        chamber_cfg:
            Chamber block from YAML.
        xy_mm:
            Candidate beam positions of shape ``(N, 2)`` in mm.

    Returns:
        Boolean mask selecting the points that lie inside the vacuum aperture.
    """
    kind = chamber_cfg["kind"].lower()
    if kind == "round":
        radius = float(chamber_cfg["radius_mm"])
        return np.sum(xy_mm**2, axis=1) < radius**2
    if kind == "ellipse":
        a = float(chamber_cfg["a_mm"])
        b = float(chamber_cfg["b_mm"])
        return (xy_mm[:, 0] / a) ** 2 + (xy_mm[:, 1] / b) ** 2 < 1.0
    poly = MplPath(close_boundary(boundary.points))
    return poly.contains_points(xy_mm)


def button_masks(button_cfg: dict[str, Any], boundary: Boundary) -> tuple[list[str], list[str], np.ndarray]:
    """Map each button to the boundary elements it covers.

    The code represents a button by the set of boundary-element midpoints lying
    within one button radius of the declared button center. This is the discrete
    counterpart of the boundary charge integration described in `README.md`
    Section 1.4.

    Args:
        button_cfg:
            The ``buttons`` block from YAML.
        boundary:
            Discrete chamber boundary.

    Returns:
        Tuple ``(labels, colors, mask_array)`` where:

        - ``labels`` is the ordered button label list
        - ``colors`` is the plotting color list
        - ``mask_array`` has shape ``(4, Nsegments)`` and selects the boundary
          elements assigned to each pickup
    """
    # Each button is represented by the subset of boundary elements whose midpoints
    # lie within one button radius of the declared button center. This is the discrete
    # approximation used later in the README Sec. 1.4 charge integration step.
    pickups = button_cfg["pickups"]
    if len(pickups) != 4:
        raise ValueError("This analysis expects exactly four BPM buttons for difference-over-sum coordinates.")

    radius = float(button_cfg["radius_mm"])
    labels: list[str] = []
    colors: list[str] = []
    masks = []
    for pickup in pickups:
        labels.append(str(pickup["label"]))
        colors.append(str(pickup.get("color", "tab:red")))
        center = np.asarray(pickup["center_mm"], dtype=float)
        mask = np.linalg.norm(boundary.midpoints - center[None, :], axis=1) <= radius
        masks.append(mask)

    mask_array = np.asarray(masks, dtype=bool)
    empty = [labels[idx] for idx, mask in enumerate(mask_array) if not np.any(mask)]
    if empty:
        raise ValueError(
            "The following buttons do not cover any boundary elements: "
            f"{empty}. Move their `center_mm` closer to the chamber boundary or increase `buttons.radius_mm`."
        )
    if np.any(mask_array.sum(axis=0) > 1):
        raise ValueError("At least one boundary element belongs to multiple buttons; reduce button radius or refine placement.")
    return labels, colors, mask_array


def build_green_matrix(boundary: Boundary, quad_order: int = 8) -> np.ndarray:
    """Assemble the boundary-element Green influence matrix.

    Physics reference:
        `README.md` Eq. (1.7) and Eq. (1.8).

    Args:
        boundary:
            Discrete chamber boundary.
        quad_order:
            Gauss-Legendre quadrature order used for off-diagonal segment
            integrals.

    Returns:
        Square matrix ``G_ij`` whose entries are the logarithmic kernel line
        integrals between boundary elements.
    """
    # README Eq. (1.7): G(r, r') = (1 / 2 pi epsilon0) ln(1 / |r - r'|).
    #
    # The common factor 1 / (2 pi epsilon0) is omitted here because it cancels in the
    # zero-potential boundary solve. We keep only the logarithmic kernel and its line
    # integral over each boundary element.
    n = len(boundary.lengths)
    g = np.empty((n, n), dtype=float)
    nodes, weights = leggauss(quad_order)

    for j in range(n):
        length_j = boundary.lengths[j]
        tangent_j = boundary.tangents[j]
        midpoint_j = boundary.midpoints[j]
        if length_j <= 0.0:
            raise ValueError("Zero-length boundary element encountered.")

        offsets = 0.5 * length_j * nodes
        quad_points = midpoint_j[None, :] + offsets[:, None] * tangent_j[None, :]
        quad_factor = 0.5 * length_j

        for i in range(n):
            if i == j:
                g[i, j] = length_j * (1.0 + math.log(2.0 / length_j))
                continue
            distances = np.linalg.norm(boundary.midpoints[i][None, :] - quad_points, axis=1)
            g[i, j] = quad_factor * np.sum(weights * np.log(1.0 / distances))
    return g


def compute_button_charges(
    beam_xy_mm: np.ndarray,
    boundary: Boundary,
    green_inv: np.ndarray,
    button_mask: np.ndarray,
) -> np.ndarray:
    """Compute induced button charge for every beam position.

    Physics reference:
        `README.md` Eq. (1.8), which gives the discrete BEM inversion for the
        induced surface charge density.

    Args:
        beam_xy_mm:
            Beam positions of shape ``(Npoints, 2)`` in mm.
        boundary:
            Discrete chamber boundary.
        green_inv:
            Precomputed inverse of the BEM influence matrix.
        button_mask:
            Boolean array of shape ``(4, Nsegments)`` describing which boundary
            elements belong to each button.

    Returns:
        Array of shape ``(4, Npoints)`` containing one induced charge per button
        for each beam position.
    """
    # README Eq. (1.8): [sigma_j] = -rho0 [G_ij]^-1 [G_i0].
    #
    # Distances from each beam point to each boundary-element midpoint define G_i0.
    # Multiplying by the precomputed inverse influence matrix yields the boundary
    # surface charge density sigma_j for every beam position.
    delta = boundary.midpoints[:, None, :] - beam_xy_mm[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    g0 = np.log(1.0 / distances)
    sigma = -(green_inv @ g0)
    dq = sigma * boundary.lengths[:, None]
    button_charge = button_mask.astype(float) @ dq
    return button_charge


def button_difference_coordinates(
    button_charge: np.ndarray,
    labels: list[str],
    layout: str = "corners",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert button charges into normalized BPM coordinates.

    Physics reference:
        `README.md` Eq. (1.9a) for ``corners`` layout and Eq. (1.9b) for
        ``cardinal`` layout.

    Args:
        button_charge:
            Button charge array of shape ``(4, Npoints)``.
        labels:
            Button labels in the same order as the first axis of ``button_charge``.
        layout:
            Either ``corners`` or ``cardinal``.

    Returns:
        Tuple ``(dx, dy, total)`` where:

        - ``dx`` is the normalized horizontal difference-over-sum value
        - ``dy`` is the normalized vertical difference-over-sum value
        - ``total`` is the total pickup signal used in the denominator
    """
    label_to_idx = {label.upper(): idx for idx, label in enumerate(labels)}
    layout_norm = layout.lower()

    if layout_norm == "corners":
        # README Eq. (1.7a): four-corner difference-over-sum for A/B/C/D.
        required = ["A", "B", "C", "D"]
        missing = [name for name in required if name not in label_to_idx]
        if missing:
            raise ValueError(f"Missing button labels for `corners` layout: {missing}")

        qa = button_charge[label_to_idx["A"]]
        qb = button_charge[label_to_idx["B"]]
        qc = button_charge[label_to_idx["C"]]
        qd = button_charge[label_to_idx["D"]]
        total = qa + qb + qc + qd
        if np.any(np.isclose(total, 0.0)):
            raise ValueError(
                "At least one beam position produced zero total pickup signal. "
                "Check button placement, chamber geometry, and beam-grid range."
            )
        dx = (qb + qc - qa - qd) / total
        dy = (qa + qb - qc - qd) / total
        return dx, dy, total

    if layout_norm == "cardinal":
        # README Eq. (1.7b): four-cardinal difference-over-sum for T/R/B/L.
        required = ["T", "R", "B", "L"]
        missing = [name for name in required if name not in label_to_idx]
        if missing:
            raise ValueError(f"Missing button labels for `cardinal` layout: {missing}")

        qt = button_charge[label_to_idx["T"]]
        qr = button_charge[label_to_idx["R"]]
        qb = button_charge[label_to_idx["B"]]
        ql = button_charge[label_to_idx["L"]]
        total = qt + qr + qb + ql
        if np.any(np.isclose(total, 0.0)):
            raise ValueError(
                "At least one beam position produced zero total pickup signal. "
                "Check button placement, chamber geometry, and beam-grid range."
            )
        dx = (qr - ql) / total
        dy = (qt - qb) / total
        return dx, dy, total

    raise ValueError(f"Unsupported buttons.layout: {layout}")


def fit_scale_factors(
    beam_xy_mm: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    fit_half_range_mm: float,
) -> tuple[float, float]:
    """Fit linear BPM scale factors near the origin.

    Physics reference:
        `README.md` Eq. (1.10), where ``X = Kx * Dx`` and ``Y = Ky * Dy``.

    Args:
        beam_xy_mm:
            True beam positions in mm.
        dx, dy:
            Normalized BPM coordinates from :func:`button_difference_coordinates`.
        fit_half_range_mm:
            Half-range around the origin used for the linear fit.

    Returns:
        Tuple ``(Kx_mm, Ky_mm)`` in mm.
    """
    # README Eq. (1.10): X = Kx * Dx, Y = Ky * Dy near the origin.
    #
    # The slopes dDx/dx and dDy/dy are fitted from the central scan lines, and Kx/Ky
    # are their inverses.
    x_mask = np.isclose(beam_xy_mm[:, 1], 0.0) & (np.abs(beam_xy_mm[:, 0]) <= fit_half_range_mm)
    y_mask = np.isclose(beam_xy_mm[:, 0], 0.0) & (np.abs(beam_xy_mm[:, 1]) <= fit_half_range_mm)
    slope_x = np.polyfit(beam_xy_mm[x_mask, 0], dx[x_mask], 1)[0]
    slope_y = np.polyfit(beam_xy_mm[y_mask, 1], dy[y_mask], 1)[0]
    return 1.0 / slope_x, 1.0 / slope_y


def polynomial_terms(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    """Construct the 2D polynomial basis used for nonlinear BPM correction.

    Args:
        x, y:
            Measured BPM coordinates.
        order:
            Maximum polynomial order in each variable.

    Returns:
        Design matrix whose columns are the basis terms ``x^i y^j``.
    """
    columns = []
    for i in range(order + 1):
        for j in range(order + 1):
            columns.append((x**i) * (y**j))
    return np.column_stack(columns)


def fit_polynomial_map(measured_xy_mm: np.ndarray, true_xy_mm: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit the nonlinear map from measured BPM coordinates to true beam position.

    Physics reference:
        `README.md` Eq. (1.11).

    Args:
        measured_xy_mm:
            Raw BPM coordinates ``(X, Y)`` in mm.
        true_xy_mm:
            True beam coordinates ``(x, y)`` in mm.
        order:
            Polynomial order used for the 2D correction map.

    Returns:
        Tuple of coefficient vectors ``(coef_x, coef_y)``.
    """
    # README Eq. (1.11): x(X,Y), y(X,Y) are represented by a 2D polynomial basis and
    # fitted in a least-squares sense from measured coordinates back to true coordinates.
    basis = polynomial_terms(measured_xy_mm[:, 0], measured_xy_mm[:, 1], order)
    coef_x, *_ = np.linalg.lstsq(basis, true_xy_mm[:, 0], rcond=None)
    coef_y, *_ = np.linalg.lstsq(basis, true_xy_mm[:, 1], rcond=None)
    return coef_x, coef_y


def apply_polynomial_map(measured_xy_mm: np.ndarray, coef_x: np.ndarray, coef_y: np.ndarray, order: int) -> np.ndarray:
    """Evaluate the fitted nonlinear BPM correction map.

    Args:
        measured_xy_mm:
            Raw BPM coordinates.
        coef_x, coef_y:
            Polynomial coefficient vectors produced by
            :func:`fit_polynomial_map`.
        order:
            Polynomial order used to construct the basis.

    Returns:
        Corrected coordinates of shape ``(Npoints, 2)`` in mm.
    """
    basis = polynomial_terms(measured_xy_mm[:, 0], measured_xy_mm[:, 1], order)
    return np.column_stack([basis @ coef_x, basis @ coef_y])


def beam_grid_points(cfg: dict[str, Any], boundary: Boundary, chamber_cfg: dict[str, Any]) -> np.ndarray:
    """Generate the rectangular transverse beam grid requested by the YAML file.

    Args:
        cfg:
            The ``beam_grid`` block from YAML.
        boundary:
            Discrete chamber boundary.
        chamber_cfg:
            Chamber block from YAML.

    Returns:
        Array of beam positions in mm that lie inside the chamber.
    """
    x_half = float(cfg["x_half_size_mm"])
    y_half = float(cfg["y_half_size_mm"])
    nx = int(cfg["nx"])
    ny = int(cfg["ny"])
    x = np.linspace(-x_half, x_half, nx)
    y = np.linspace(-y_half, y_half, ny)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return points[inside_chamber(boundary, chamber_cfg, points)]


def build_line_density(bunch_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Build the normalized longitudinal line-charge density on a uniform z grid.

    Physics reference:
        `README.md` Eq. (1.2) for charge normalization.

    Args:
        bunch_cfg:
            The ``bunch`` block from YAML, or a compatible override dictionary.

    Returns:
        Tuple ``(z_m, line_density_c_per_m)``.
    """
    grid_cfg = bunch_cfg.get("longitudinal_grid", {})
    grid_number = int(grid_cfg.get("grid_number", 200))
    if grid_number <= 0:
        raise ValueError("`bunch.longitudinal_grid.grid_number` must be a positive integer.")
    charge_c = float(bunch_cfg["charge_nC"]) * 1e-9
    density_cfg = bunch_cfg["density"]
    kind = density_cfg["kind"].lower()

    if kind == "gaussian":
        sigma_mm = float(density_cfg["sigma_mm"])
        dz_mm = sigma_mm / grid_number
        cutoff_sigma = float(density_cfg.get("cutoff_sigma", 0.0))
        if cutoff_sigma > 0.0:
            z_max = cutoff_sigma * sigma_mm
        else:
            z_max = float(grid_cfg.get("no_cut_span_sigma", 8.0)) * sigma_mm
        z_mm = np.arange(-z_max, z_max + dz_mm, dz_mm)
        profile = np.exp(-0.5 * (z_mm / sigma_mm) ** 2)
    elif kind == "array":
        samples = density_cfg["samples"]
        sample_z, sample_i = parse_density_samples(samples)
        span_mm = sample_z.max() - sample_z.min()
        dz_mm = span_mm / grid_number if span_mm > 0.0 else 1.0
        z_mm = np.arange(sample_z.min(), sample_z.max() + dz_mm, dz_mm)
        order_request = int(density_cfg.get("interpolation_order", 5))
        spline_order = min(order_request, len(sample_z) - 1)
        if spline_order < 1:
            raise ValueError("At least two longitudinal density sample points are required.")
        if spline_order == 1:
            interp = interp1d(sample_z, sample_i, kind="linear", bounds_error=False, fill_value=0.0)
            profile = interp(z_mm)
        else:
            spline = InterpolatedUnivariateSpline(sample_z, sample_i, k=spline_order)
            profile = spline(z_mm)
            outside = (z_mm < sample_z.min()) | (z_mm > sample_z.max())
            profile[outside] = 0.0
        profile = np.clip(profile, 0.0, None)
    else:
        raise ValueError(f"Unsupported density kind: {kind}")

    norm = np.trapz(profile, z_mm * 1e-3)
    if norm <= 0.0:
        raise ValueError("Beam density normalization failed; the longitudinal profile integrates to zero.")
    # README Eq. (1.2): the user-defined shape is normalized first, then multiplied by
    # the physical bunch charge so that the integral of line density equals Q_bunch.
    line_density = charge_c * profile / norm
    return z_mm * 1e-3, line_density


def parse_density_samples(samples: Any) -> tuple[np.ndarray, np.ndarray]:
    """Parse arbitrary longitudinal density samples from any supported YAML form.

    Supported formats are documented in `README.md` Section 2.2.

    Args:
        samples:
            Either a compact string, list of ``[z, value]`` pairs, or list of
            ``{z_mm, peakcurrent}`` mappings.

    Returns:
        Tuple ``(sample_z_mm, sample_values)`` sorted by increasing ``z`` with
        duplicate ``z`` values removed.
    """
    if isinstance(samples, str):
        matches = re.findall(
            r"\{\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\}",
            samples,
        )
        if not matches:
            raise ValueError(
                "Could not parse `bunch.density.samples`. "
                "Expected compact syntax like \"{{-120,0.15},{-60,0.8},{0,1},{60,0.8},{120,0.15}}\"."
            )
        sample_z = np.asarray([float(z) for z, _ in matches], dtype=float)
        sample_i = np.asarray([float(i) for _, i in matches], dtype=float)
    elif isinstance(samples, list):
        if not samples:
            raise ValueError("`bunch.density.samples` is empty.")
        if isinstance(samples[0], dict):
            sample_z = np.asarray([item["z_mm"] for item in samples], dtype=float)
            sample_i = np.asarray([item["peakcurrent"] for item in samples], dtype=float)
        else:
            arr = np.asarray(samples, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(
                    "`bunch.density.samples` list form must be [[z1, i1], [z2, i2], ...]."
                )
            sample_z = arr[:, 0]
            sample_i = arr[:, 1]
    else:
        raise ValueError(
            "`bunch.density.samples` must be a compact string, a list of [z, peakcurrent] pairs, "
            "or a list of {z_mm, peakcurrent} maps."
        )

    order = np.argsort(sample_z)
    sample_z = sample_z[order]
    sample_i = sample_i[order]

    unique_z, unique_idx = np.unique(sample_z, return_index=True)
    sample_z = unique_z
    sample_i = sample_i[unique_idx]

    if len(sample_z) < 2:
        raise ValueError("At least two distinct z positions are required in `bunch.density.samples`.")
    return sample_z, sample_i


def button_width_kernel(z_m: np.ndarray, radius_mm: float) -> np.ndarray:
    """Evaluate the longitudinal button-width kernel.

    Physics reference:
        `README.md` Eq. (1.1): ``w(z) = 2 * sqrt(r_b^2 - z^2)``.

    Args:
        z_m:
            Longitudinal coordinate array in meters.
        radius_mm:
            Button radius in mm.

    Returns:
        Width kernel array in meters.
    """
    # README Eq. (1.1): w(z) = 2 * sqrt(r_b^2 - z^2) inside the button projection.
    radius_m = radius_mm * 1e-3
    kernel = np.zeros_like(z_m)
    mask = np.abs(z_m) <= radius_m
    kernel[mask] = 2.0 * np.sqrt(np.maximum(radius_m**2 - z_m[mask] ** 2, 0.0))
    return kernel


def nearest_pow2(n: int) -> int:
    """Return the smallest power of two greater than or equal to ``n``."""
    return 1 << int(math.ceil(math.log2(max(2, n))))


def narrowest_filter_feature_hz(filters: list[dict[str, Any]]) -> float | None:
    """Return the narrowest frequency feature that must be resolved in Fig. 7.

    Args:
        filters:
            List of filter dictionaries from ``filter.comparison_filters``.

    Returns:
        Smallest useful bandwidth/cutoff in Hz, or ``None`` if the filters do
        not define a frequency scale.
    """
    features: list[float] = []
    for filt in filters:
        filt_type = str(filt.get("type", "none")).lower()
        if filt_type == "bandpass_butter" and "bandwidth_hz" in filt:
            features.append(float(filt["bandwidth_hz"]))
        elif filt_type in {"lowpass_butter", "lowpass_bessel"} and "cutoff_hz" in filt:
            features.append(float(filt["cutoff_hz"]))
    return min(features) if features else None


def spectrum_fft_size_for_filters(n_signal: int, dt: float, filters: list[dict[str, Any]]) -> int:
    """Choose a zero-padded FFT size fine enough for narrow filter plots.

    The BAR Fig. 7 comparison includes a 20 MHz-wide BPF centered at 500 MHz.
    With the long 62 mm bunch, the natural FFT grid can be tens of MHz per bin,
    which misses the BPF passband peak and makes ``|H_BPF|`` appear much smaller
    than the intended transfer function.  This helper makes the Fig. 7 spectrum
    grid substantially finer than the narrowest cutoff/bandwidth while leaving
    the time-domain signal model unchanged.

    Args:
        n_signal:
            Number of time-domain samples in the voltage waveform.
        dt:
            Time step in seconds.
        filters:
            Comparison filters to plot.

    Returns:
        Power-of-two FFT length.
    """
    base_fft = nearest_pow2(max(n_signal, 2))
    feature_hz = narrowest_filter_feature_hz(filters)
    if feature_hz is None or feature_hz <= 0.0:
        return base_fft
    target_df_hz = feature_hz / 40.0
    required = int(math.ceil(1.0 / (dt * target_df_hz)))
    return nearest_pow2(max(base_fft, required))


def spectrum_fft_size_for_button_impedance(n_signal: int, dt: float, z0: float, capacitance_f: float) -> int:
    """Choose a Fig. 3 FFT size that resolves the button RC roll-off.

    Args:
        n_signal:
            Number of time-domain samples in the image-current waveform.
        dt:
            Time step in seconds.
        z0:
            Cable/reference impedance in ohms.
        capacitance_f:
            Button capacitance in farads.

    Returns:
        Power-of-two FFT length. The value is capped to keep very short bunch
        cases practical while still extending far beyond the old fixed 10 GHz
        plotting range.
    """
    base_fft = nearest_pow2(max(n_signal, 2))
    if z0 <= 0.0 or capacitance_f <= 0.0:
        return base_fft
    rc_hz = 1.0 / (2.0 * math.pi * z0 * capacitance_f)
    target_df_hz = rc_hz / 40.0
    required = int(math.ceil(1.0 / (dt * target_df_hz)))
    return min(nearest_pow2(max(base_fft, required)), 4_194_304)


def auto_log_frequency_xlim(
    freqs_hz: np.ndarray,
    curves: list[np.ndarray],
    rel_floor: float = 1e-3,
    pad_decades: float = 0.12,
) -> tuple[float, float]:
    """Choose log-frequency axis limits from plotted curve amplitudes.

    Args:
        freqs_hz:
            Frequency samples in Hz.
        curves:
            Curves plotted against ``freqs_hz``. Each curve is normalized by
            its own peak before thresholding, so voltage, impedance, and filter
            transfer curves can be mixed.
        rel_floor:
            Relative level that defines the useful plotted bandwidth. A value of
            ``1e-3`` means the axis extends until each curve has dropped to about
            0.1% of its own peak, which is a practical "near noise floor" for
            these diagnostic plots.
        pad_decades:
            Small log-space padding added around the active frequency range.

    Returns:
        ``(f_min_ghz, f_max_ghz)`` suitable for ``set_xlim``.
    """
    freqs = np.asarray(freqs_hz, dtype=float)
    positive = np.isfinite(freqs) & (freqs > 0.0)
    if not np.any(positive):
        return 1e-3, 1.0

    active = np.zeros_like(freqs, dtype=bool)
    for curve in curves:
        values = np.abs(np.asarray(curve, dtype=float))
        valid = positive & np.isfinite(values) & (values > 0.0)
        if not np.any(valid):
            continue
        peak = float(np.nanmax(values[valid]))
        if peak <= 0.0:
            continue
        active |= valid & (values >= rel_floor * peak)

    if not np.any(active):
        active = positive

    active_freqs = freqs[active]
    available_freqs = freqs[positive]
    log_low = math.log10(float(active_freqs.min())) - pad_decades
    log_high = math.log10(float(active_freqs.max())) + pad_decades
    low_hz = max(float(available_freqs.min()), 10.0**log_low)
    high_hz = min(float(available_freqs.max()), 10.0**log_high)
    if high_hz <= low_hz:
        high_hz = min(float(available_freqs.max()), low_hz * 10.0)
    return low_hz * 1e-9, high_hz * 1e-9


def configured_frequency_xlim(
    range_cfg: Any,
    figure_key: str,
    auto_xlim: tuple[float, float],
) -> tuple[float, float]:
    """Resolve YAML frequency-axis settings for Fig. 3/Fig. 7.

    Accepted YAML forms:

    - ``frequency_range: auto``
    - ``frequency_range: {min_ghz: 0.01, max_ghz: 10}``
    - ``frequency_range: {figure3: auto, figure7: {min_ghz: 0.01, max_ghz: 10}}``
    - ``frequency_range: [0.01, 10]``

    Args:
        range_cfg:
            Parsed YAML value from ``filter.frequency_range``.
        figure_key:
            Either ``figure3`` or ``figure7``.
        auto_xlim:
            Automatically calculated ``(min_ghz, max_ghz)`` fallback.

    Returns:
        Frequency x-limits in GHz.
    """
    selected = range_cfg
    if selected is None:
        return auto_xlim
    if isinstance(selected, str):
        if selected.lower() == "auto":
            return auto_xlim
        raise ValueError("`filter.frequency_range` must be `auto`, [min_ghz, max_ghz], or a mapping.")
    if isinstance(selected, (list, tuple)):
        if len(selected) != 2:
            raise ValueError("List-style `filter.frequency_range` must contain exactly [min_ghz, max_ghz].")
        low, high = float(selected[0]), float(selected[1])
    elif isinstance(selected, dict):
        if figure_key in selected:
            return configured_frequency_xlim(selected[figure_key], figure_key, auto_xlim)
        if str(selected.get("mode", "")).lower() == "auto":
            return auto_xlim
        if "min_ghz" not in selected or "max_ghz" not in selected:
            return auto_xlim
        low, high = float(selected["min_ghz"]), float(selected["max_ghz"])
    else:
        raise ValueError("`filter.frequency_range` must be `auto`, [min_ghz, max_ghz], or a mapping.")

    if low <= 0.0 or high <= low:
        raise ValueError("Frequency range must satisfy 0 < min_ghz < max_ghz.")
    return low, high


def configured_time_xlim_ns(range_cfg: Any, auto_xlim: tuple[float, float]) -> tuple[float, float]:
    """Resolve YAML time-axis settings for Fig. 5.

    Accepted YAML forms:

    - ``figure5_time_range: auto``
    - ``figure5_time_range: {min_ns: 0.0, max_ns: 2.0}``
    - ``figure5_time_range: [0.0, 2.0]``

    Args:
        range_cfg:
            Parsed YAML value from ``filter.figure5_time_range``.
        auto_xlim:
            Automatically calculated ``(min_ns, max_ns)`` fallback.

    Returns:
        Time x-limits in ns.
    """
    if range_cfg is None:
        return auto_xlim
    if isinstance(range_cfg, str):
        if range_cfg.lower() == "auto":
            return auto_xlim
        raise ValueError("`filter.figure5_time_range` must be `auto`, [min_ns, max_ns], or a mapping.")
    if isinstance(range_cfg, (list, tuple)):
        if len(range_cfg) != 2:
            raise ValueError("List-style `filter.figure5_time_range` must contain exactly [min_ns, max_ns].")
        low, high = float(range_cfg[0]), float(range_cfg[1])
    elif isinstance(range_cfg, dict):
        if str(range_cfg.get("mode", "")).lower() == "auto":
            return auto_xlim
        if "min_ns" not in range_cfg or "max_ns" not in range_cfg:
            return auto_xlim
        low, high = float(range_cfg["min_ns"]), float(range_cfg["max_ns"])
    else:
        raise ValueError("`filter.figure5_time_range` must be `auto`, [min_ns, max_ns], or a mapping.")
    if high <= low:
        raise ValueError("Fig. 5 time range must satisfy min_ns < max_ns.")
    return low, high


def apply_frequency_response(signal_t: np.ndarray, response: np.ndarray) -> np.ndarray:
    """Apply a real frequency-domain transfer function to a time signal.

    Args:
        signal_t:
            Real-valued time-domain signal.
        response:
            Real or complex one-sided FFT response sampled on the target
            frequency grid.

    Returns:
        Filtered time-domain signal with the same length as the input.
    """
    n_fft = 2 * (len(response) - 1)
    spectrum = np.fft.rfft(signal_t, n=n_fft)
    filtered = np.fft.irfft(spectrum * response, n=n_fft)
    return filtered[: len(signal_t)]


def image_charge_denominator_m(boundary: Boundary, cfg: dict[str, Any]) -> float:
    """Return the geometric denominator for longitudinal image charge.

    The BAR/SLAC button formula uses ``2*pi*b`` in the denominator.  For a
    non-round chamber this value is best supplied explicitly from the paper's
    chamber dimension.  If no value is supplied, the code falls back to the
    chamber perimeter used by the earlier generic model.

    Args:
        boundary:
            Discrete chamber boundary, used for the fallback perimeter.
        cfg:
            Full YAML configuration.

    Returns:
        Denominator in meters.
    """
    signal_model = cfg.get("signal_model", {})
    if "image_charge_denominator_mm" in signal_model:
        return float(signal_model["image_charge_denominator_mm"]) * 1e-3
    return boundary.perimeter * 1e-3


def cable_attenuation_fc_hz(cable_cfg: dict[str, Any]) -> float:
    """Resolve the skin-effect cable attenuation frequency ``fc``.

    The BAR note defines ``fc`` as the frequency at which the cable amplitude is
    attenuated by a factor of ``e``:

        |H(f)| = exp(-sqrt(f / fc)).

    If ``attenuation_fc_hz`` is supplied directly, that value is used. Otherwise
    the code derives ``fc`` from physical cable data:

        total_loss_dB = attenuation_db_per_100m * length_m / 100
        sqrt(f_ref / fc) = ln(10) * total_loss_dB / 20

    Args:
        cable_cfg:
            ``filter.cable`` block from YAML.

    Returns:
        Effective ``fc`` in Hz, or ``0`` if the cable is disabled.
    """
    if not cable_cfg.get("enabled", False):
        return 0.0
    if "attenuation_fc_hz" in cable_cfg:
        return float(cable_cfg["attenuation_fc_hz"])

    try:
        length_m = float(cable_cfg["length_m"])
        match_frequency_hz = float(cable_cfg.get("match_frequency_hz", cable_cfg.get("reference_frequency_hz")))
        attenuation_db_per_100m = float(cable_cfg["attenuation_db_per_100m"])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Cable attenuation requires either `attenuation_fc_hz` or all of "
            "`length_m`, `match_frequency_hz`, and `attenuation_db_per_100m`."
        ) from exc

    total_loss_db = attenuation_db_per_100m * length_m / 100.0
    exponent_at_match = math.log(10.0) * total_loss_db / 20.0
    if match_frequency_hz <= 0.0 or exponent_at_match <= 0.0:
        raise ValueError("Cable length, match frequency, and attenuation must be positive.")
    return match_frequency_hz / (exponent_at_match**2)


def cable_transfer_response(freqs_hz: np.ndarray, cable_cfg: dict[str, Any]) -> np.ndarray:
    """Evaluate the complex skin-effect cable transfer response.

    Args:
        freqs_hz:
            One-sided FFT frequency grid in Hz.
        cable_cfg:
            ``filter.cable`` block from YAML.

    Returns:
        Complex cable response sampled on ``freqs_hz``.
    """
    response = np.ones_like(freqs_hz, dtype=complex)
    if cable_cfg.get("enabled", False):
        fc_hz = cable_attenuation_fc_hz(cable_cfg)
        sqrt_loss = np.sqrt(np.maximum(freqs_hz, 0.0) / fc_hz)
        include_phase = bool(cable_cfg.get("skin_effect_phase", True))
        # README Eq. (1.6): the skin-effect cable model is
        # H_cable(f) = exp(-(1 + i) * sqrt(f / fc)).  The real part gives the
        # attenuation and the imaginary part gives the dispersive delay.
        phase_factor = 1.0j if include_phase else 0.0
        response *= np.exp(-(1.0 + phase_factor) * sqrt_loss)
    return response


def cable_tail_time_s(signal_data: dict[str, np.ndarray | float]) -> float:
    """Choose a zero-padded time tail for Fig. 5 cable-input/output plots."""
    z0 = float(signal_data["characteristic_impedance_ohm"])
    capacitance_f = float(signal_data["button_capacitance_f"])
    rc_tail_s = 20.0 * z0 * capacitance_f
    fc_hz = float(signal_data.get("cable_attenuation_fc_hz", 0.0))
    cable_tail_s = 8.0 / fc_hz if fc_hz > 0.0 else 0.0
    return max(rc_tail_s, cable_tail_s)


def cable_io_waveforms_for_plot(signal_data: dict[str, np.ndarray | float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Fig. 5 input/output waveforms with enough zero-padded tail.

    ``build_signal_base`` keeps arrays on the bunch sampling window used by the
    rest of the analysis. Fig. 5 needs a longer time axis for short bunches so
    the button RC response and the cable output can decay back near zero.
    """
    image_current = np.asarray(signal_data["image_current_a"])
    t_s = np.asarray(signal_data["t_s"])
    dt = float(signal_data["dt_s"])
    z0 = float(signal_data["characteristic_impedance_ohm"])
    capacitance_f = float(signal_data["button_capacitance_f"])
    tail_samples = int(math.ceil(cable_tail_time_s(signal_data) / dt))
    n_time = len(image_current) + max(0, tail_samples)
    n_fft = min(nearest_pow2(n_time), 4_194_304)
    freqs_hz = np.fft.rfftfreq(n_fft, d=dt)
    omega = 2.0 * math.pi * freqs_hz
    current_spec = np.fft.rfft(image_current, n=n_fft)
    z_button = 1.0 / (1.0 / z0 + 1j * omega * capacitance_f)
    cable_cfg = {
        "enabled": bool(signal_data.get("cable_enabled", False)),
        "attenuation_fc_hz": float(signal_data.get("cable_attenuation_fc_hz", 1.0)),
        "skin_effect_phase": bool(signal_data.get("cable_skin_effect_phase", True)),
    }
    cable_response = cable_transfer_response(freqs_hz, cable_cfg)
    v_button = np.fft.irfft(current_spec * z_button, n=n_fft)
    v_cable = np.fft.irfft(current_spec * z_button * cable_response, n=n_fft)
    t_plot_s = float(t_s[0]) + np.arange(n_fft, dtype=float) * dt
    return t_plot_s, v_button, v_cable


def build_signal_base(
    boundary: Boundary,
    cfg: dict[str, Any],
    bunch_override: dict[str, Any] | None = None,
) -> dict[str, np.ndarray | float]:
    """Build the signal chain up to the cable output, before analog filtering.

    Physics reference:
        - `README.md` Eq. (1.2): line-density normalization
        - `README.md` Eq. (1.3): image charge convolution
        - `README.md` Eq. (1.4): image current derivative
        - `README.md` Eq. (1.5): button impedance
        - `README.md` Eq. (1.6): skin-effect cable transfer

    Args:
        boundary:
            Discrete chamber boundary.
        cfg:
            Full YAML configuration.
        bunch_override:
            Optional bunch block used for comparison cases such as Fig. 9.

    Returns:
        Dictionary containing longitudinal arrays, spectra, impedance, and cable
        output needed by later plotting functions.
    """
    bunch_cfg = resolve_default_bunch_cfg(cfg)
    if bunch_override:
        bunch_cfg.update(bunch_override)

    z_m, line_density = build_line_density(bunch_cfg)
    dz = float(np.mean(np.diff(z_m)))
    current_profile_a = constants.c * line_density
    mean_z_m = float(np.trapz(z_m * line_density, z_m) / np.trapz(line_density, z_m))
    sigma_z_m = math.sqrt(
        max(
            0.0,
            float(np.trapz(((z_m - mean_z_m) ** 2) * line_density, z_m) / np.trapz(line_density, z_m)),
        )
    )
    # README Eq. (1.2): q_img(t) is the convolution of the line-charge density with
    # the button-width kernel, divided by the geometric image-charge denominator.
    width = button_width_kernel(z_m, float(cfg["buttons"]["radius_mm"]))
    image_denominator_m = image_charge_denominator_m(boundary, cfg)
    image_charge = np.convolve(line_density, width, mode="same") * dz / image_denominator_m
    t_s = z_m / constants.c
    # README Eq. (1.4): I_img(t) = d q_img(t) / d t.
    image_current = np.gradient(image_charge, t_s)

    button_cfg = cfg["buttons"]
    z0 = float(cfg["filter"].get("characteristic_impedance_ohm", 50.0))
    capacitance_pf = button_cfg.get("capacitance_pf")
    if capacitance_pf is None:
        gap_mm = float(button_cfg["gap_mm"])
        thickness_mm = float(button_cfg["thickness_mm"])
        radius_m = float(button_cfg["radius_mm"]) * 1e-3
        gap_m = gap_mm * 1e-3
        thickness_m = thickness_mm * 1e-3
        capacitance_f = 2.0 * math.pi * constants.epsilon_0 * thickness_m / math.log(1.0 + gap_m / radius_m)
    else:
        capacitance_f = float(capacitance_pf) * 1e-12

    n_fft = max(16384, nearest_pow2(len(t_s)))
    dt = float(np.mean(np.diff(t_s)))
    freqs = np.fft.rfftfreq(n_fft, d=dt)
    omega = 2.0 * math.pi * freqs
    current_spec = np.fft.rfft(image_current, n=n_fft)
    # README Eq. (1.5): Z_b(omega) = 1 / (1/Z0 + i omega C_b).
    z_button = 1.0 / (1.0 / z0 + 1j * omega * capacitance_f)
    v_button = np.fft.irfft(current_spec * z_button, n=n_fft)[: len(t_s)]

    cable_cfg = cfg["filter"].get("cable", {})
    effective_cable_fc_hz = cable_attenuation_fc_hz(cable_cfg)
    response = cable_transfer_response(freqs, cable_cfg)
    v_cable = apply_frequency_response(v_button, response)

    return {
        "z_m": z_m,
        "t_s": t_s,
        "dt_s": dt,
        "freqs_hz": freqs,
        "line_density_cpm": line_density,
        "current_profile_a": current_profile_a,
        "button_width_m": width,
        "image_charge_c": image_charge,
        "image_current_a": image_current,
        "image_current_fft_abs": np.abs(current_spec),
        "button_voltage_v": v_button,
        "button_voltage_fft_abs": np.abs(np.fft.rfft(v_button, n=n_fft)),
        "cable_voltage_v": v_cable,
        "cable_voltage_fft_abs": np.abs(np.fft.rfft(v_cable, n=n_fft)),
        "button_impedance_ohm_abs": np.abs(z_button),
        "button_capacitance_f": capacitance_f,
        "characteristic_impedance_ohm": z0,
        "cable_transfer_abs": np.abs(response),
        "cable_enabled": bool(cable_cfg.get("enabled", False)),
        "cable_attenuation_fc_hz": effective_cable_fc_hz,
        "cable_skin_effect_phase": bool(cable_cfg.get("skin_effect_phase", True)),
        "image_charge_denominator_m": image_denominator_m,
        "charge_nC": float(bunch_cfg["charge_nC"]),
        "sigma_z_m": sigma_z_m,
        "button_radius_mm": float(cfg["buttons"]["radius_mm"]),
    }


def resolve_default_bunch_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve the default bunch definition used by the main signal chain.

    Rule:
        If ``signal_cases`` exists and is non-empty, the first entry provides the
        default ``charge_nC`` and ``density``. The shared
        ``bunch.longitudinal_grid`` block still comes from the top-level
        ``bunch`` section. If ``signal_cases`` is absent, the full top-level
        ``bunch`` block is used directly.

    Args:
        cfg:
            Full YAML configuration.

    Returns:
        Effective bunch configuration dictionary.
    """
    bunch_cfg = copy.deepcopy(cfg["bunch"])
    signal_cases = cfg.get("signal_cases", [])
    if signal_cases:
        first_case = signal_cases[0]
        bunch_cfg["charge_nC"] = first_case["charge_nC"]
        bunch_cfg["density"] = copy.deepcopy(first_case["density"])
    return bunch_cfg


def build_analog_sos(dt: float, analog_cfg: dict[str, Any]) -> np.ndarray | None:
    """Create the Butterworth SOS representation for one analog filter.

    Args:
        dt:
            Time step in seconds.
        analog_cfg:
            Filter definition from YAML.

    Returns:
        Second-order-sections array, or ``None`` for ``type: none``.
    """
    analog_type = analog_cfg.get("type", "none").lower()
    if analog_type == "none":
        return None
    if analog_type == "lowpass_butter":
        cutoff_hz = float(analog_cfg["cutoff_hz"])
        order = int(analog_cfg.get("order", 4))
        return butter(order, cutoff_hz, btype="lowpass", fs=1.0 / dt, output="sos")
    if analog_type == "lowpass_bessel":
        cutoff_hz = float(analog_cfg["cutoff_hz"])
        order = int(analog_cfg.get("order", 4))
        return bessel(order, cutoff_hz, btype="lowpass", fs=1.0 / dt, norm="phase", output="sos")
    if analog_type == "bandpass_butter":
        center_hz = float(analog_cfg["center_hz"])
        bandwidth_hz = float(analog_cfg["bandwidth_hz"])
        order = int(analog_cfg.get("order", 4))
        low = center_hz - 0.5 * bandwidth_hz
        high = center_hz + 0.5 * bandwidth_hz
        return butter(order, [low, high], btype="bandpass", fs=1.0 / dt, output="sos")
    raise ValueError(f"Unsupported analog filter type: {analog_type}")


def apply_analog_filter(v_cable: np.ndarray, dt: float, analog_cfg: dict[str, Any]) -> np.ndarray:
    """Apply the configured analog filter to the cable-output waveform.

    Args:
        v_cable:
            Cable-output signal in volts.
        dt:
            Time step in seconds.
        analog_cfg:
            Filter configuration block.

    Returns:
        Filtered voltage waveform.
    """
    v_filtered, _ = apply_analog_filter_causal(v_cable, dt, analog_cfg)
    return v_filtered[: len(v_cable)]


def filter_tail_time_s(analog_cfg: dict[str, Any]) -> float:
    """Choose a practical zero-padding tail for causal filter plots."""
    analog_type = analog_cfg.get("type", "none").lower()
    if analog_type == "bandpass_butter":
        return 5.0 / float(analog_cfg["bandwidth_hz"])
    if analog_type in {"lowpass_butter", "lowpass_bessel"}:
        return 8.0 / float(analog_cfg["cutoff_hz"])
    return 0.0


def apply_analog_filter_causal(
    v_cable: np.ndarray,
    dt: float,
    analog_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one causal analog filter pass and keep enough zero tail to decay.

    The paper's Fig. 8 is a causal time-domain response: the filter output is zero
    before the bunch-driven signal reaches the electronics, then rings and decays.
    This function starts the waveform at the first sampled cable point, appends
    zeros, and applies a one-pass SOS filter.
    """
    sos = build_analog_sos(dt, analog_cfg)
    tail_samples = int(math.ceil(filter_tail_time_s(analog_cfg) / dt))
    padded = np.pad(v_cable, (0, max(0, tail_samples)))
    t_s = np.arange(len(padded), dtype=float) * dt
    if sos is None:
        return padded, t_s
    return sosfilt(sos, padded), t_s


def analog_transfer_abs(freqs_hz: np.ndarray, dt: float, analog_cfg: dict[str, Any]) -> np.ndarray:
    """Evaluate the analog-filter amplitude response on a chosen frequency grid.

    Args:
        freqs_hz:
            Frequency samples in Hz.
        dt:
            Time step in seconds.
        analog_cfg:
            Filter configuration block.

    Returns:
        Absolute transfer-function magnitude ``|H(f)|``.
    """
    sos = build_analog_sos(dt, analog_cfg)
    if sos is None:
        return np.ones_like(freqs_hz)
    _, h = sosfreqz(sos, worN=freqs_hz, fs=1.0 / dt)
    return np.abs(h)


def analog_transfer_complex(freqs_hz: np.ndarray, dt: float, analog_cfg: dict[str, Any]) -> np.ndarray:
    """Evaluate the complex one-pass analog transfer function H(f).

    This is used to reproduce the paper-style filtering operation:

        Vf(t) = IFFT(H(f) * FFT(Vc(t)))

    rather than a forward-backward time-domain filter.

    Args:
        freqs_hz:
            Frequency samples in Hz.
        dt:
            Time step in seconds.
        analog_cfg:
            Filter configuration block.

    Returns:
        Complex transfer-function samples.
    """
    sos = build_analog_sos(dt, analog_cfg)
    if sos is None:
        return np.ones_like(freqs_hz, dtype=complex)
    _, h = sosfreqz(sos, worN=freqs_hz, fs=1.0 / dt)
    return h


def signal_chain(boundary: Boundary, cfg: dict[str, Any]) -> dict[str, np.ndarray | float]:
    """Build the full default signal chain including the primary analog filter.

    Args:
        boundary:
            Discrete chamber boundary.
        cfg:
            Full YAML configuration.

    Returns:
        Dictionary containing the base signal-chain data plus the field
        ``filtered_voltage_v`` for the main configured analog filter.
    """
    base = build_signal_base(boundary, cfg)
    analog_cfg = cfg["filter"].get("analog", {"type": "none"})
    v_filtered, t_filtered = apply_analog_filter_causal(np.asarray(base["cable_voltage_v"]), float(base["dt_s"]), analog_cfg)
    return {**base, "filtered_voltage_v": v_filtered, "filtered_t_s": t_filtered}


def resolution_curves(kx_mm: float, ky_mm: float, resolution_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the BPM resolution curves versus relative voltage error.

    Physics reference:
        `README.md` Eq. (1.12).

    Args:
        kx_mm, ky_mm:
            BPM scale factors in mm.
        resolution_cfg:
            ``filter.resolution`` block from YAML.

    Returns:
        Tuple ``(relative_error, sigma_x_mm, sigma_y_mm)``.
    """
    # README Eq. (1.12): sigma_x ~= Kx * sigma_V / (2V), sigma_y ~= Ky * sigma_V / (2V).
    rel = np.logspace(
        math.log10(float(resolution_cfg.get("relative_error_min", 1e-4))),
        math.log10(float(resolution_cfg.get("relative_error_max", 1e-2))),
        int(resolution_cfg.get("num_points", 200)),
    )
    sigma_x = 0.5 * kx_mm * rel
    sigma_y = 0.5 * ky_mm * rel
    return rel, sigma_x, sigma_y


def plot_boundary(ax: plt.Axes, boundary: Boundary, button_masks_arr: np.ndarray, button_colors: list[str]) -> None:
    """Draw the chamber boundary and highlight button-covered boundary segments.

    Args:
        ax:
            Target Matplotlib axes.
        boundary:
            Discrete chamber boundary.
        button_masks_arr:
            Boolean boundary-element masks for each button.
        button_colors:
            Display color for each button.
    """
    closed = close_boundary(boundary.points)
    ax.plot(closed[:, 0], closed[:, 1], color="black", linewidth=1.5, zorder=1)
    for mask, color in zip(button_masks_arr, button_colors):
        starts = boundary.seg_start[mask]
        ends = boundary.seg_end[mask]
        for start, end in zip(starts, ends):
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=3.2, solid_capstyle="round", zorder=2)


def plot_linearity(
    output_path: Path,
    boundary: Boundary,
    button_masks_arr: np.ndarray,
    button_colors: list[str],
    true_xy_mm: np.ndarray,
    measured_xy_mm: np.ndarray,
) -> None:
    """Plot the Fig. 11 style raw BPM linearity map.

    Args:
        output_path:
            Target image file.
        boundary:
            Chamber boundary to draw.
        button_masks_arr, button_colors:
            Button geometry overlays.
        true_xy_mm:
            True input beam grid in mm.
        measured_xy_mm:
            Raw linear BPM reconstruction in mm.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    plot_boundary(ax, boundary, button_masks_arr, button_colors)
    ax.scatter(true_xy_mm[:, 0], true_xy_mm[:, 1], s=8, color="royalblue", marker="s", linewidths=0.0, label="input")
    ax.scatter(measured_xy_mm[:, 0], measured_xy_mm[:, 1], s=10, color="crimson", marker=".", linewidths=0.0, label="measured")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_polyfit(
    output_path: Path,
    boundary: Boundary,
    button_masks_arr: np.ndarray,
    button_colors: list[str],
    true_xy_mm: np.ndarray,
    fit_xy_mm: np.ndarray,
) -> None:
    """Plot the Fig. 12 style polynomial-corrected BPM map."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    plot_boundary(ax, boundary, button_masks_arr, button_colors)
    ax.scatter(true_xy_mm[:, 0], true_xy_mm[:, 1], s=8, color="royalblue", marker="s", linewidths=0.0, label="input")
    ax.scatter(fit_xy_mm[:, 0], fit_xy_mm[:, 1], s=9, color="crimson", marker=".", linewidths=0.0, label="polynomial fit")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_resolution(
    output_path: Path,
    rel: np.ndarray,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
    signal_data: dict[str, np.ndarray | float],
) -> None:
    """Plot the Fig. 13 style BPM resolution curves."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    fig.suptitle(signal_title(signal_data))
    ax.loglog(rel, sigma_x, color="royalblue", linewidth=2.0, linestyle="--", label="hor")
    ax.loglog(rel, sigma_y, color="crimson", linewidth=1.8, linestyle=(0, (5, 2)), label="ver")
    ax.set_xlabel(r"$\sigma_V / V$")
    ax.set_ylabel(r"$\sigma_x, \sigma_y$ (mm)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def signal_title(signal_data: dict[str, np.ndarray | float]) -> str:
    """Format the default bunch charge and RMS length for figure titles."""
    charge_nC = float(signal_data["charge_nC"])
    sigma_mm = 1e3 * float(signal_data["sigma_z_m"])
    return f"Charge = {charge_nC:.3g} nC, Beam sigma = {sigma_mm:.3g} mm"


def plot_signal_summary(
    output_path: Path,
    signal_data: dict[str, np.ndarray | float],
    comparison_filters: list[dict[str, Any]],
) -> None:
    """Generate a compact three-panel summary of the longitudinal signal chain.

    The first panel shows the bunch current profile and button width. The second
    panel keeps the short image-current/button-voltage time scale. The third
    panel is dedicated to filtered voltage so the LPF/BPF amplitudes are readable
    even when they are much smaller than the raw button voltage.
    """
    z_mm = np.asarray(signal_data["z_m"]) * 1e3
    t_s = np.asarray(signal_data["t_s"])
    t_ns = (t_s - t_s[0]) * 1e9
    dt = float(signal_data["dt_s"])
    current_profile_a = np.asarray(signal_data["current_profile_a"])
    image_current_ma = 1e3 * np.asarray(signal_data["image_current_a"])
    v_button = np.asarray(signal_data["button_voltage_v"])
    v_cable = np.asarray(signal_data["cable_voltage_v"])
    button_width_mm = 1e3 * np.asarray(signal_data["button_width_m"])
    charge_nC = float(signal_data["charge_nC"])
    sigma_mm = 1e3 * float(signal_data["sigma_z_m"])
    button_radius_mm = float(signal_data["button_radius_mm"])

    z_half_span_mm = max(2.0 * button_radius_mm, 5.0 * sigma_mm)
    t_half_span_ns = 1e9 * (z_half_span_mm * 1e-3) / constants.c

    fig, axes = plt.subplots(3, 1, figsize=(8.4, 8.2), sharex=False)
    fig.suptitle(f"Charge = {charge_nC:.3g} nC, Beam sigma = {sigma_mm:.3g} mm")

    ax_top = axes[0]
    ax_top_r = ax_top.twinx()
    ax_top.plot(z_mm, current_profile_a, color="tab:blue", linewidth=1.8, label="current profile [A]")
    ax_top_r.plot(z_mm, button_width_mm, color="tab:orange", linewidth=1.8, label="button width (mm)")
    ax_top.set_xlim(-z_half_span_mm, z_half_span_mm)
    ax_top.set_xlabel("z (mm)")
    ax_top.set_ylabel("Current (A)")
    ax_top_r.set_ylabel("Button width (mm)")
    ax_top.grid(True, alpha=0.3)
    top_lines = ax_top.get_lines() + ax_top_r.get_lines()
    ax_top.legend(top_lines, [line.get_label() for line in top_lines], loc="upper right")

    ax_bot = axes[1]
    ax_bot_r = ax_bot.twinx()
    ax_bot.plot(t_ns, image_current_ma, color="tab:green", linewidth=1.8, label="image current (mA)")
    ax_bot_r.plot(t_ns, v_button, color="tab:orange", linewidth=1.8, label="button voltage (V)")
    raw_min_end_ns = max(0.05, 1e9 * (2.0 * button_radius_mm * 1e-3) / constants.c)
    raw_plot_end_ns = max(
        post_peak_decay_time_ns(t_ns, image_current_ma, remaining_fraction=0.01, min_end_ns=raw_min_end_ns),
        post_peak_decay_time_ns(t_ns, v_button, remaining_fraction=0.01, min_end_ns=raw_min_end_ns),
    )
    ax_bot.set_xlim(0.0, raw_plot_end_ns)
    ax_bot.margins(x=0.0)
    ax_bot.set_xlabel("t (ns)")
    ax_bot.set_ylabel("Image current (mA)")
    ax_bot_r.set_ylabel("Voltage (V)")
    ax_bot.grid(True, alpha=0.3)
    bot_lines = ax_bot.get_lines() + ax_bot_r.get_lines()
    ax_bot.legend(bot_lines, [line.get_label() for line in bot_lines], loc="upper right")

    ax_filter = axes[2]
    filter_plot_end_ns = 0.0
    if not comparison_filters:
        ax_filter.plot(
            np.asarray(signal_data["filtered_t_s"]) * 1e9,
            np.asarray(signal_data["filtered_voltage_v"]),
            linewidth=1.8,
            label="filtered voltage",
        )
        filter_plot_end_ns = post_peak_decay_time_ns(
            np.asarray(signal_data["filtered_t_s"]) * 1e9,
            np.asarray(signal_data["filtered_voltage_v"]),
            remaining_fraction=0.05,
            min_end_ns=2.0 * t_half_span_ns,
        )
    else:
        for filt in comparison_filters:
            vf, tf_s = apply_analog_filter_causal(v_cable, dt, filt)
            tf_ns = tf_s * 1e9
            filter_plot_end_ns = max(
                filter_plot_end_ns,
                post_peak_decay_time_ns(tf_ns, vf, remaining_fraction=0.05, min_end_ns=5.0),
            )
            ax_filter.plot(tf_ns, vf, linewidth=1.8, label=str(filt.get("name", filt.get("type", "filter"))))
    ax_filter.set_xlim(0.0, filter_plot_end_ns)
    ax_filter.margins(x=0.0)
    ax_filter.set_xlabel("t (ns)")
    ax_filter.set_ylabel("Filtered voltage (V)")
    ax_filter.grid(True, alpha=0.3)
    ax_filter.legend(loc="upper right")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def single_sided_spectrum_mV(signal: np.ndarray, n_fft: int) -> np.ndarray:
    """Return a single-sided FFT amplitude spectrum in mV."""
    spectrum = np.abs(np.fft.rfft(signal, n=n_fft)) / n_fft
    if len(spectrum) > 2:
        spectrum[1:-1] *= 2.0
    return 1e3 * spectrum


def quiet_time_ns(t_ns: np.ndarray, signal: np.ndarray, rel_threshold: float = 0.01, min_end_ns: float = 0.0) -> float:
    """Choose a plotting endpoint after a signal has decayed near zero."""
    abs_signal = np.abs(signal)
    peak = float(abs_signal.max()) if len(abs_signal) else 0.0
    if peak <= 0.0:
        return float(max(t_ns[-1], min_end_ns))
    active = np.flatnonzero(abs_signal > rel_threshold * peak)
    if len(active) == 0:
        return float(max(t_ns[-1], min_end_ns))
    idx = min(len(t_ns) - 1, active[-1] + max(5, len(t_ns) // 100))
    return float(max(t_ns[idx], min_end_ns))


def post_peak_decay_time_ns(
    t_ns: np.ndarray,
    signal: np.ndarray,
    remaining_fraction: float = 0.05,
    min_end_ns: float = 5.0,
) -> float:
    """Return a plot endpoint after the waveform decays from its peak.

    Args:
        t_ns:
            Monotonic time array in ns, normally starting at zero for Fig. 8.
        signal:
            Voltage waveform.
        remaining_fraction:
            Fraction of the peak magnitude that defines "nearly damped away".
            The default 0.05 means the signal has dropped by 95%.
        min_end_ns:
            Minimum endpoint in ns.

    Returns:
        Time in ns after the last post-peak sample above the threshold, with a
        small visual pad. The value is clamped to the available sampled tail.
    """
    if len(t_ns) == 0:
        return min_end_ns
    abs_signal = np.abs(signal)
    peak = float(abs_signal.max()) if len(abs_signal) else 0.0
    if peak <= 0.0:
        return float(max(t_ns[-1], min_end_ns))
    peak_idx = int(np.argmax(abs_signal))
    threshold = remaining_fraction * peak
    post_peak_active = np.flatnonzero(abs_signal[peak_idx:] >= threshold)
    if len(post_peak_active) == 0:
        end_idx = peak_idx
    else:
        end_idx = peak_idx + int(post_peak_active[-1])
    pad_ns = max(2.0, 0.05 * max(float(t_ns[end_idx]), min_end_ns))
    return float(min(max(t_ns[-1], min_end_ns), max(float(t_ns[end_idx]) + pad_ns, min_end_ns)))


def plot_fig3_impedance_current_spectrum(
    output_path: Path,
    signal_data: dict[str, np.ndarray | float],
    frequency_range_cfg: Any = "auto",
) -> None:
    """Plot BAR-like Fig. 3: button impedance and FFT magnitude of image current."""
    dt = float(signal_data["dt_s"])
    image_current = np.asarray(signal_data["image_current_a"])
    z0 = float(signal_data["characteristic_impedance_ohm"])
    capacitance_f = float(signal_data["button_capacitance_f"])
    n_fft = spectrum_fft_size_for_button_impedance(len(image_current), dt, z0, capacitance_f)
    freqs_hz = np.fft.rfftfreq(n_fft, d=dt)
    f_ghz = freqs_hz * 1e-9
    omega = 2.0 * math.pi * freqs_hz
    z_abs = np.abs(1.0 / (1.0 / z0 + 1j * omega * capacitance_f))
    i_abs = 1e3 * np.abs(np.fft.rfft(image_current, n=n_fft))

    fig, ax1 = plt.subplots(figsize=(6.8, 4.8))
    fig.suptitle(signal_title(signal_data))
    ax2 = ax1.twinx()
    ax1.loglog(f_ghz[1:], z_abs[1:], color="tab:blue", linewidth=1.8, label="impedance")
    ax2.loglog(f_ghz[1:], i_abs[1:], color="tab:red", linewidth=1.8, label="current")
    ax1.set_xlabel("f (GHz)")
    ax1.set_ylabel("|Zb| (ohm)")
    ax2.set_ylabel("|I| (mA)")
    auto_xlim = auto_log_frequency_xlim(freqs_hz, [z_abs, i_abs], rel_floor=1e-3)
    ax1.set_xlim(*configured_frequency_xlim(frequency_range_cfg, "figure3", auto_xlim))
    ax1.grid(True, which="both", alpha=0.3)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="lower left")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_fig4_current_voltage(output_path: Path, signal_data: dict[str, np.ndarray | float]) -> None:
    """Plot BAR-like Fig. 4: image current and button voltage versus time."""
    t_ns = np.asarray(signal_data["t_s"]) * 1e9
    i_ma = 1e3 * np.asarray(signal_data["image_current_a"])
    v_b = np.asarray(signal_data["button_voltage_v"])

    fig, ax1 = plt.subplots(figsize=(6.8, 4.8))
    fig.suptitle(signal_title(signal_data))
    ax2 = ax1.twinx()
    ax1.plot(t_ns, i_ma, color="tab:blue", linewidth=1.8, label="image current")
    ax2.plot(t_ns, v_b, color="tab:red", linewidth=1.8, label="button voltage")
    ax1.set_xlabel("t (ns)")
    ax1.set_ylabel("Iimg (mA)")
    ax2.set_ylabel("Vb (V)")
    ax1.grid(True, alpha=0.3)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper right")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_fig5_cable_io(
    output_path: Path,
    signal_data: dict[str, np.ndarray | float],
    time_range_cfg: Any = "auto",
) -> None:
    """Plot BAR-like Fig. 5: button voltage at cable input and output."""
    t_plot_s, v_b, v_c = cable_io_waveforms_for_plot(signal_data)
    t_ns = (t_plot_s - t_plot_s[0]) * 1e9

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    fig.suptitle(signal_title(signal_data))
    ax.plot(t_ns, v_b, color="tab:blue", linewidth=1.8, label="input")
    ax.plot(t_ns, v_c, color="tab:red", linewidth=1.8, label="output")
    x_end_ns = max(
        post_peak_decay_time_ns(t_ns, v_b, remaining_fraction=0.05, min_end_ns=0.1),
        post_peak_decay_time_ns(t_ns, v_c, remaining_fraction=0.05, min_end_ns=0.1),
    )
    ax.set_xlim(*configured_time_xlim_ns(time_range_cfg, (0.0, x_end_ns)))
    ax.margins(x=0.0)
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("Vb, Vc (V)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_fig7_frequency_filters(
    output_path: Path,
    signal_data: dict[str, np.ndarray | float],
    comparison_filters: list[dict[str, Any]],
    frequency_range_cfg: Any = "auto",
) -> None:
    """Plot BAR-like Fig. 7 for one or more comparison filters.

    Each subplot shows:
    - FFT amplitude of the button voltage
    - FFT amplitude after the cable
    - FFT amplitude after one comparison filter
    - the filter amplitude response ``|H|``
    """
    if not comparison_filters:
        return
    dt = float(signal_data["dt_s"])
    v_b = np.asarray(signal_data["button_voltage_v"])
    v_cable = np.asarray(signal_data["cable_voltage_v"])
    # README Section 1.3 and BAR Fig. 7: use a plotting FFT grid fine enough
    # to resolve the narrow BPF bandwidth before evaluating H(f) * FFT(Vc).
    n_fft = spectrum_fft_size_for_filters(len(v_cable), dt, comparison_filters)
    freqs_hz = np.fft.rfftfreq(n_fft, d=dt)
    f_ghz = freqs_hz * 1e-9
    v_b_fft = single_sided_spectrum_mV(v_b, n_fft)
    v_c_fft = single_sided_spectrum_mV(v_cable, n_fft)

    ncols = len(comparison_filters)
    fig, axes = plt.subplots(1, ncols, figsize=(6.6 * ncols, 4.8), squeeze=False)
    fig.suptitle(signal_title(signal_data))
    for ax, filt in zip(axes[0], comparison_filters):
        h = analog_transfer_complex(freqs_hz, dt, filt)
        vf_fft = 1e3 * np.abs(np.fft.rfft(v_cable, n=n_fft) * h) / n_fft
        if len(vf_fft) > 2:
            vf_fft[1:-1] *= 2.0
        h_abs = analog_transfer_abs(freqs_hz, dt, filt)
        ax2 = ax.twinx()
        ax.loglog(f_ghz[1:], v_b_fft[1:], color="tab:blue", linewidth=1.6, label="Vb")
        ax.loglog(f_ghz[1:], v_c_fft[1:], color="tab:orange", linewidth=1.6, label="Vc")
        ax.loglog(f_ghz[1:], vf_fft[1:], color="tab:red", linewidth=1.6, label=str(filt.get("name", "Vf")))
        ax2.semilogx(f_ghz[1:], h_abs[1:], color="tab:green", linewidth=1.6, label="|H|")
        ax.set_xlabel("f (GHz)")
        ax.set_ylabel("V (mV)")
        ax2.set_ylabel("|H|")
        auto_xlim = auto_log_frequency_xlim(freqs_hz, [v_b_fft, v_c_fft, vf_fft, h_abs], rel_floor=1e-3)
        ax.set_xlim(*configured_frequency_xlim(frequency_range_cfg, "figure7", auto_xlim))
        ax.set_ylim(bottom=1e-4)
        ax2.set_ylim(0.0, max(1.05, 1.05 * float(np.nanmax(h_abs))))
        ax.set_title(str(filt.get("name", filt.get("type", "filter"))))
        ax.grid(True, which="both", alpha=0.3)
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], loc="lower left")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_fig8_filter_outputs(
    output_path: Path,
    signal_data: dict[str, np.ndarray | float],
    comparison_filters: list[dict[str, Any]],
) -> None:
    """Plot BAR-like Fig. 8: time-domain outputs of comparison filters."""
    if not comparison_filters:
        return
    dt = float(signal_data["dt_s"])
    v_cable = np.asarray(signal_data["cable_voltage_v"])

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    fig.suptitle(signal_title(signal_data))
    x_end_ns = 0.0
    for filt in comparison_filters:
        vf, t_s = apply_analog_filter_causal(v_cable, dt, filt)
        t_ns = t_s * 1e9
        x_end_ns = max(
            x_end_ns,
            post_peak_decay_time_ns(t_ns, vf, remaining_fraction=0.05, min_end_ns=5.0),
        )
        ax.plot(t_ns, vf, linewidth=1.8, label=str(filt.get("name", filt.get("type", "filter"))))
    ax.set_xlim(0.0, x_end_ns)
    ax.margins(x=0.0)
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("Vf (V)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_fig9_button_voltage_cases(
    output_path: Path,
    boundary: Boundary,
    cfg: dict[str, Any],
    signal_cases: list[dict[str, Any]],
) -> None:
    """Plot BAR-like Fig. 9 for multiple charge/length comparison cases.

    Args:
        output_path:
            Target image file.
        boundary:
            Chamber boundary used by the signal model.
        cfg:
            Full YAML configuration.
        signal_cases:
            List of case dictionaries, each containing a name, charge, and
            density block override.
    """
    if not signal_cases:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for case in signal_cases:
        case_bunch = {
            "charge_nC": case["charge_nC"],
            "density": case["density"],
            "longitudinal_grid": cfg["bunch"].get("longitudinal_grid", {}),
        }
        data = build_signal_base(boundary, cfg, bunch_override=case_bunch)
        ax.plot(np.asarray(data["t_s"]) * 1e9, np.asarray(data["button_voltage_v"]), linewidth=1.7, label=str(case["name"]))
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("Vb (V)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def residual_metrics(reference_xy_mm: np.ndarray, estimate_xy_mm: np.ndarray) -> tuple[float, float]:
    """Compute RMS and max position error between two point clouds.

    Args:
        reference_xy_mm:
            Ground-truth coordinates in mm.
        estimate_xy_mm:
            Reconstructed coordinates in mm.

    Returns:
        Tuple ``(rms_error_mm, max_error_mm)``.
    """
    err = np.linalg.norm(estimate_xy_mm - reference_xy_mm, axis=1)
    return float(np.sqrt(np.mean(err**2))), float(np.max(err))


def write_report(
    output_path: Path,
    cfg_path: Path,
    cfg: dict[str, Any],
    boundary: Boundary,
    kx_mm: float,
    ky_mm: float,
    linear_rms_mm: float,
    linear_max_mm: float,
    poly_rms_mm: float,
    poly_max_mm: float,
    signal_data: dict[str, np.ndarray | float],
    reference_rel_error: float,
) -> None:
    """Write the Markdown summary report for one completed run.

    Args:
        output_path:
            Destination Markdown file.
        cfg_path:
            YAML path used for the run.
        cfg:
            Full parsed YAML configuration.
        boundary:
            Chamber boundary description.
        kx_mm, ky_mm:
            BPM scale factors.
        linear_rms_mm, linear_max_mm:
            Raw BPM reconstruction errors.
        poly_rms_mm, poly_max_mm:
            Polynomial-corrected reconstruction errors.
        signal_data:
            Signal-chain results from :func:`signal_chain`.
        reference_rel_error:
            Reference ``sigma_V / V`` point for the resolution summary.
    """
    cfg_filter = cfg.get("filter", {})
    v_filtered = np.asarray(signal_data["filtered_voltage_v"])
    v_button = np.asarray(signal_data["button_voltage_v"])
    v_cable = np.asarray(signal_data["cable_voltage_v"])
    v_peak = float(np.max(np.abs(v_filtered)))
    v_rms = float(np.sqrt(np.mean(v_filtered**2)))
    cap_pf = float(signal_data["button_capacitance_f"]) * 1e12
    z0 = float(signal_data["characteristic_impedance_ohm"])
    image_denominator_mm = float(signal_data["image_charge_denominator_m"]) * 1e3
    cable_fc_hz = float(signal_data["cable_attenuation_fc_hz"])

    sigma_x_ref_mm = 0.5 * kx_mm * reference_rel_error
    sigma_y_ref_mm = 0.5 * ky_mm * reference_rel_error

    notes = []
    if cfg["chamber"]["kind"].lower() == "polygon":
        notes.append(
            "The sample polygon was taken from the BAR note figures because `BeampositionMonitor.gdf` was not found in the local Research tree."
        )
    effective_bunch = resolve_default_bunch_cfg(cfg)
    if effective_bunch["density"]["kind"].lower() == "gaussian" and float(effective_bunch["density"].get("cutoff_sigma", 0.0)) == 0.0:
        notes.append("The uncapped Gaussian is evaluated on a finite numerical window set by `bunch.longitudinal_grid.no_cut_span_sigma`.")

    lines = [
        "# BAR BPM Analysis Report",
        "",
        f"- Config: `{cfg_path}`",
        f"- Chamber type: `{cfg['chamber']['kind']}`",
        f"- Boundary perimeter: {boundary.perimeter:.3f} mm",
        f"- Longitudinal image-charge denominator: {image_denominator_mm:.3f} mm",
        f"- Button capacitance used in signal model: {cap_pf:.3f} pF",
        f"- Characteristic impedance: {z0:.1f} ohm",
        f"- Effective cable attenuation frequency `fc`: {cable_fc_hz:.3e} Hz",
        "",
        "## Linearity",
        "",
        f"- Linear scale factor `Kx`: {kx_mm:.3f} mm",
        f"- Linear scale factor `Ky`: {ky_mm:.3f} mm",
        f"- RMS position error before polynomial correction: {linear_rms_mm * 1e3:.2f} um",
        f"- Max position error before polynomial correction: {linear_max_mm:.3f} mm",
        f"- RMS position error after polynomial correction: {poly_rms_mm * 1e3:.2f} um",
        f"- Max position error after polynomial correction: {poly_max_mm:.3f} mm",
        "",
        "## Signal Summary",
        "",
        f"- Peak voltage at button output: {np.max(np.abs(v_button)):.3e} V",
        f"- Peak voltage after cable model: {np.max(np.abs(v_cable)):.3e} V",
        f"- Peak voltage after analog filter: {v_peak:.3e} V",
        f"- RMS voltage after analog filter: {v_rms:.3e} V",
        "",
        "## Resolution",
        "",
        f"- Reference relative voltage error `sigma_V / V`: {reference_rel_error:.6g}",
        f"- Estimated horizontal resolution at the reference point: {sigma_x_ref_mm * 1e3:.2f} um",
        f"- Estimated vertical resolution at the reference point: {sigma_y_ref_mm * 1e3:.2f} um",
        "",
    ]

    if notes:
        lines.extend(["## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")
        if cfg_filter.get("analog", {}).get("type", "").lower() == "bandpass_butter" and v_peak < 1e-6:
            lines.append("- The configured band-pass filter suppresses most of the long-bunch spectrum, so the filtered time-domain voltage is very small.")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(cfg_path: Path) -> dict[str, Path]:
    """Execute one complete BPM analysis run from a YAML file.

    This is the main orchestration function for the script. It loads the config,
    builds the chamber model, runs the BEM solve, computes the signal chain,
    generates all enabled figures, writes the report, and returns the output file
    paths.

    Args:
        cfg_path:
            Path to the YAML input file.

    Returns:
        Dictionary mapping logical output names to generated file paths.
    """
    cfg = load_config(cfg_path)
    output_dir = Path(cfg.get("output", {}).get("directory", "outputs/default"))
    if not output_dir.is_absolute():
        output_dir = cfg_path.parent / output_dir
    ensure_dir(output_dir)

    boundary = build_boundary(cfg["chamber"])
    labels, colors, masks = button_masks(cfg["buttons"], boundary)
    green = build_green_matrix(boundary, quad_order=int(cfg["chamber"].get("quadrature_order", 8)))
    green_inv = np.linalg.inv(green)

    beam_xy_mm = beam_grid_points(cfg["beam_grid"], boundary, cfg["chamber"])
    button_charge = compute_button_charges(beam_xy_mm, boundary, green_inv, masks)
    dx, dy, _total = button_difference_coordinates(
        button_charge,
        labels,
        layout=str(cfg["buttons"].get("layout", "corners")),
    )

    fit_half = float(cfg["beam_grid"].get("linear_fit_half_range_mm", 5.0))
    kx_mm, ky_mm = fit_scale_factors(beam_xy_mm, dx, dy, fit_half)
    measured_xy_mm = np.column_stack([kx_mm * dx, ky_mm * dy])

    poly_order = int(cfg["beam_grid"].get("polynomial_order", 5))
    coef_x, coef_y = fit_polynomial_map(measured_xy_mm, beam_xy_mm, poly_order)
    fit_xy_mm = apply_polynomial_map(measured_xy_mm, coef_x, coef_y, poly_order)

    linear_rms_mm, linear_max_mm = residual_metrics(beam_xy_mm, measured_xy_mm)
    poly_rms_mm, poly_max_mm = residual_metrics(beam_xy_mm, fit_xy_mm)

    signal_data = signal_chain(boundary, cfg)
    filter_cfg = cfg.get("filter", {})
    comparison_filters = filter_cfg.get("comparison_filters", [])
    frequency_range_cfg = filter_cfg.get("frequency_range", "auto")
    figure5_time_range_cfg = filter_cfg.get("figure5_time_range", "auto")
    signal_cases = cfg.get("signal_cases", [])
    resolution_cfg = cfg.get("filter", {}).get("resolution", {})
    rel, sigma_x, sigma_y = resolution_curves(kx_mm, ky_mm, resolution_cfg)
    reference_rel_error = float(resolution_cfg.get("reference_relative_error", 1e-3))

    linearity_path = output_dir / "figure_11_linearity.png"
    polyfit_path = output_dir / "figure_12_polyfit.png"
    resolution_path = output_dir / "figure_13_resolution.png"
    signal_path = output_dir / "signal_summary.png"
    fig3_path = output_dir / "figure_3_impedance_and_image_current_fft.png"
    fig4_path = output_dir / "figure_4_image_current_and_button_voltage.png"
    fig5_path = output_dir / "figure_5_cable_input_output.png"
    fig7_path = output_dir / "figure_7_signal_fft_and_filter_response.png"
    fig8_path = output_dir / "figure_8_filter_output_voltage.png"
    fig9_path = output_dir / "figure_9_button_voltage_cases.png"
    report_path = output_dir / "analysis_report.md"

    plot_linearity(linearity_path, boundary, masks, colors, beam_xy_mm, measured_xy_mm)
    plot_polyfit(polyfit_path, boundary, masks, colors, beam_xy_mm, fit_xy_mm)
    plot_resolution(resolution_path, rel, sigma_x, sigma_y, signal_data)
    plot_signal_summary(signal_path, signal_data, comparison_filters)
    plot_fig3_impedance_current_spectrum(fig3_path, signal_data, frequency_range_cfg)
    plot_fig4_current_voltage(fig4_path, signal_data)
    plot_fig5_cable_io(fig5_path, signal_data, figure5_time_range_cfg)
    plot_fig7_frequency_filters(fig7_path, signal_data, comparison_filters, frequency_range_cfg)
    plot_fig8_filter_outputs(fig8_path, signal_data, comparison_filters)
    plot_fig9_button_voltage_cases(fig9_path, boundary, cfg, signal_cases)
    write_report(
        report_path,
        cfg_path,
        cfg,
        boundary,
        kx_mm,
        ky_mm,
        linear_rms_mm,
        linear_max_mm,
        poly_rms_mm,
        poly_max_mm,
        signal_data,
        reference_rel_error,
    )

    return {
        "linearity": linearity_path,
        "polyfit": polyfit_path,
        "resolution": resolution_path,
        "signal_summary": signal_path,
        "fig3": fig3_path,
        "fig4": fig4_path,
        "fig5": fig5_path,
        "fig7": fig7_path,
        "fig8": fig8_path,
        "fig9": fig9_path,
        "report": report_path,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="BAR BPM boundary-element analysis and figure generation.")
    parser.add_argument("config", type=Path, help="Path to the YAML input file.")
    args = parser.parse_args()
    outputs = run(args.config.resolve())
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
