#!/usr/bin/env python3
from __future__ import annotations
"""
BAR BPM analysis tool.

Maintenance note:
- Keep the physics comments in this file aligned with the equations documented in
  README.md, especially the labeled equations in Section 1.
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
from scipy.signal import butter, sosfiltfilt, sosfreqz

matplotlib.use("Agg")


@dataclass
class Boundary:
    points: np.ndarray
    seg_start: np.ndarray
    seg_end: np.ndarray
    midpoints: np.ndarray
    tangents: np.ndarray
    lengths: np.ndarray
    perimeter: float
    kind: str


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def mm_to_m(values: Any) -> Any:
    return np.asarray(values, dtype=float) * 1e-3


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def close_boundary(points: np.ndarray) -> np.ndarray:
    if np.allclose(points[0], points[-1]):
        return points.copy()
    return np.vstack([points, points[0]])


def sample_polygon(points_mm: list[list[float]], n_segments: int) -> np.ndarray:
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
    # README Eq. (1.5): G(r, r') = (1 / 2 pi epsilon0) ln(1 / |r - r'|).
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
    # README Eq. (1.6): [sigma_j] = -rho0 [G_ij]^-1 [G_i0].
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
    # README Eq. (1.8): X = Kx * Dx, Y = Ky * Dy near the origin.
    #
    # The slopes dDx/dx and dDy/dy are fitted from the central scan lines, and Kx/Ky
    # are their inverses.
    x_mask = np.isclose(beam_xy_mm[:, 1], 0.0) & (np.abs(beam_xy_mm[:, 0]) <= fit_half_range_mm)
    y_mask = np.isclose(beam_xy_mm[:, 0], 0.0) & (np.abs(beam_xy_mm[:, 1]) <= fit_half_range_mm)
    slope_x = np.polyfit(beam_xy_mm[x_mask, 0], dx[x_mask], 1)[0]
    slope_y = np.polyfit(beam_xy_mm[y_mask, 1], dy[y_mask], 1)[0]
    return 1.0 / slope_x, 1.0 / slope_y


def polynomial_terms(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    columns = []
    for i in range(order + 1):
        for j in range(order + 1):
            columns.append((x**i) * (y**j))
    return np.column_stack(columns)


def fit_polynomial_map(measured_xy_mm: np.ndarray, true_xy_mm: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    # README Eq. (1.9): x(X,Y), y(X,Y) are represented by a 2D polynomial basis and
    # fitted in a least-squares sense from measured coordinates back to true coordinates.
    basis = polynomial_terms(measured_xy_mm[:, 0], measured_xy_mm[:, 1], order)
    coef_x, *_ = np.linalg.lstsq(basis, true_xy_mm[:, 0], rcond=None)
    coef_y, *_ = np.linalg.lstsq(basis, true_xy_mm[:, 1], rcond=None)
    return coef_x, coef_y


def apply_polynomial_map(measured_xy_mm: np.ndarray, coef_x: np.ndarray, coef_y: np.ndarray, order: int) -> np.ndarray:
    basis = polynomial_terms(measured_xy_mm[:, 0], measured_xy_mm[:, 1], order)
    return np.column_stack([basis @ coef_x, basis @ coef_y])


def beam_grid_points(cfg: dict[str, Any], boundary: Boundary, chamber_cfg: dict[str, Any]) -> np.ndarray:
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
    grid_cfg = bunch_cfg.get("longitudinal_grid", {})
    dz_mm = float(grid_cfg.get("dz_mm", 0.25))
    charge_c = float(bunch_cfg["charge_nC"]) * 1e-9
    density_cfg = bunch_cfg["density"]
    kind = density_cfg["kind"].lower()

    if kind == "gaussian":
        sigma_mm = float(density_cfg["sigma_mm"])
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
    # README Eq. (1.1): w(z) = 2 * sqrt(r_b^2 - z^2) inside the button projection.
    radius_m = radius_mm * 1e-3
    kernel = np.zeros_like(z_m)
    mask = np.abs(z_m) <= radius_m
    kernel[mask] = 2.0 * np.sqrt(np.maximum(radius_m**2 - z_m[mask] ** 2, 0.0))
    return kernel


def nearest_pow2(n: int) -> int:
    return 1 << int(math.ceil(math.log2(max(2, n))))


def apply_frequency_response(signal_t: np.ndarray, response: np.ndarray) -> np.ndarray:
    n_fft = 2 * (len(response) - 1)
    spectrum = np.fft.rfft(signal_t, n=n_fft)
    filtered = np.fft.irfft(spectrum * response, n=n_fft)
    return filtered[: len(signal_t)]


def build_signal_base(
    boundary: Boundary,
    cfg: dict[str, Any],
    bunch_override: dict[str, Any] | None = None,
) -> dict[str, np.ndarray | float]:
    bunch_cfg = copy.deepcopy(cfg["bunch"])
    if bunch_override:
        bunch_cfg.update(bunch_override)

    z_m, line_density = build_line_density(bunch_cfg)
    dz = float(np.mean(np.diff(z_m)))
    # README Eq. (1.3): q_img(t) is the convolution of the line-charge density with
    # the button-width kernel, scaled here by the chamber perimeter fraction.
    width = button_width_kernel(z_m, float(cfg["buttons"]["radius_mm"]))
    image_charge = np.convolve(line_density, width, mode="same") * dz / boundary.perimeter
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
    response = np.ones_like(freqs)
    if cable_cfg.get("enabled", False):
        fc_hz = float(cable_cfg["attenuation_fc_hz"])
        response *= np.exp(-np.sqrt(np.maximum(freqs, 0.0) / fc_hz))
    v_cable = apply_frequency_response(v_button, response)

    return {
        "z_m": z_m,
        "t_s": t_s,
        "dt_s": dt,
        "freqs_hz": freqs,
        "line_density_cpm": line_density,
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
        "cable_transfer_abs": response,
    }


def build_analog_sos(dt: float, analog_cfg: dict[str, Any]) -> np.ndarray | None:
    analog_type = analog_cfg.get("type", "none").lower()
    if analog_type == "none":
        return None
    if analog_type == "lowpass_butter":
        cutoff_hz = float(analog_cfg["cutoff_hz"])
        order = int(analog_cfg.get("order", 4))
        return butter(order, cutoff_hz, btype="lowpass", fs=1.0 / dt, output="sos")
    if analog_type == "bandpass_butter":
        center_hz = float(analog_cfg["center_hz"])
        bandwidth_hz = float(analog_cfg["bandwidth_hz"])
        order = int(analog_cfg.get("order", 4))
        low = center_hz - 0.5 * bandwidth_hz
        high = center_hz + 0.5 * bandwidth_hz
        return butter(order, [low, high], btype="bandpass", fs=1.0 / dt, output="sos")
    raise ValueError(f"Unsupported analog filter type: {analog_type}")


def apply_analog_filter(v_cable: np.ndarray, dt: float, analog_cfg: dict[str, Any]) -> np.ndarray:
    sos = build_analog_sos(dt, analog_cfg)
    if sos is None:
        return v_cable
    return sosfiltfilt(sos, v_cable)


def analog_transfer_abs(freqs_hz: np.ndarray, dt: float, analog_cfg: dict[str, Any]) -> np.ndarray:
    sos = build_analog_sos(dt, analog_cfg)
    if sos is None:
        return np.ones_like(freqs_hz)
    _, h = sosfreqz(sos, worN=freqs_hz, fs=1.0 / dt)
    return np.abs(h)


def signal_chain(boundary: Boundary, cfg: dict[str, Any]) -> dict[str, np.ndarray | float]:
    base = build_signal_base(boundary, cfg)
    analog_cfg = cfg["filter"].get("analog", {"type": "none"})
    v_filtered = apply_analog_filter(np.asarray(base["cable_voltage_v"]), float(base["dt_s"]), analog_cfg)
    return {**base, "filtered_voltage_v": v_filtered}


def resolution_curves(kx_mm: float, ky_mm: float, resolution_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # README Eq. (1.10): sigma_x ~= Kx * sigma_V / (2V), sigma_y ~= Ky * sigma_V / (2V).
    rel = np.logspace(
        math.log10(float(resolution_cfg.get("relative_error_min", 1e-4))),
        math.log10(float(resolution_cfg.get("relative_error_max", 1e-2))),
        int(resolution_cfg.get("num_points", 200)),
    )
    sigma_x = 0.5 * kx_mm * rel
    sigma_y = 0.5 * ky_mm * rel
    return rel, sigma_x, sigma_y


def plot_boundary(ax: plt.Axes, boundary: Boundary, button_masks_arr: np.ndarray, button_colors: list[str]) -> None:
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


def plot_resolution(output_path: Path, rel: np.ndarray, sigma_x: np.ndarray, sigma_y: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.loglog(rel, sigma_x, color="royalblue", linewidth=1.8, label="hor")
    ax.loglog(rel, sigma_y, color="crimson", linewidth=1.8, label="ver")
    ax.set_xlabel(r"$\sigma_V / V$")
    ax.set_ylabel(r"$\sigma_x, \sigma_y$ (mm)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_signal_summary(output_path: Path, signal_data: dict[str, np.ndarray | float]) -> None:
    z_mm = np.asarray(signal_data["z_m"]) * 1e3
    t_ns = np.asarray(signal_data["t_s"]) * 1e9
    image_current_ma = 1e3 * np.asarray(signal_data["image_current_a"])
    v_button = np.asarray(signal_data["button_voltage_v"])
    v_filtered = np.asarray(signal_data["filtered_voltage_v"])

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=False)
    axes[0].plot(z_mm, 1e9 * np.asarray(signal_data["line_density_cpm"]) * 1e-3, label="line density (scaled)")
    axes[0].plot(z_mm, 1e3 * np.asarray(signal_data["button_width_m"]), label="button width (mm)")
    axes[0].set_xlabel("z (mm)")
    axes[0].set_ylabel("arb.")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t_ns, image_current_ma, color="tab:green", label="image current (mA)")
    axes[1].plot(t_ns, v_button, color="tab:orange", label="button voltage (V)")
    axes[1].plot(t_ns, v_filtered, color="tab:red", label="filtered voltage (V)")
    axes[1].set_xlabel("t (ns)")
    axes[1].set_ylabel("signal")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_fig3_impedance_current_spectrum(output_path: Path, signal_data: dict[str, np.ndarray | float]) -> None:
    f_ghz = np.asarray(signal_data["freqs_hz"]) * 1e-9
    z_abs = np.asarray(signal_data["button_impedance_ohm_abs"])
    i_abs = 1e3 * np.asarray(signal_data["image_current_fft_abs"])

    fig, ax1 = plt.subplots(figsize=(6.8, 4.8))
    ax2 = ax1.twinx()
    ax1.loglog(f_ghz[1:], z_abs[1:], color="tab:blue", linewidth=1.8, label="impedance")
    ax2.loglog(f_ghz[1:], i_abs[1:], color="tab:red", linewidth=1.8, label="current")
    ax1.set_xlabel("f (GHz)")
    ax1.set_ylabel("|Zb| (ohm)")
    ax2.set_ylabel("|I| (mA)")
    ax1.set_xlim(1e-2, 1e1)
    ax1.grid(True, which="both", alpha=0.3)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_fig4_current_voltage(output_path: Path, signal_data: dict[str, np.ndarray | float]) -> None:
    t_ns = np.asarray(signal_data["t_s"]) * 1e9
    i_ma = 1e3 * np.asarray(signal_data["image_current_a"])
    v_b = np.asarray(signal_data["button_voltage_v"])

    fig, ax1 = plt.subplots(figsize=(6.8, 4.8))
    ax2 = ax1.twinx()
    ax1.plot(t_ns, i_ma, color="tab:blue", linewidth=1.8, label="image current")
    ax2.plot(t_ns, v_b, color="tab:red", linewidth=1.8, label="button voltage")
    ax1.set_xlabel("t (ns)")
    ax1.set_ylabel("Iimg (mA)")
    ax2.set_ylabel("Vb (V)")
    ax1.grid(True, alpha=0.3)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_fig5_cable_io(output_path: Path, signal_data: dict[str, np.ndarray | float]) -> None:
    t_ns = np.asarray(signal_data["t_s"]) * 1e9
    v_b = np.asarray(signal_data["button_voltage_v"])
    v_c = np.asarray(signal_data["cable_voltage_v"])

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(t_ns, v_b, color="tab:blue", linewidth=1.8, label="input")
    ax.plot(t_ns, v_c, color="tab:red", linewidth=1.8, label="output")
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("Vb, Vc (V)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_fig7_frequency_filters(
    output_path: Path,
    signal_data: dict[str, np.ndarray | float],
    comparison_filters: list[dict[str, Any]],
) -> None:
    if not comparison_filters:
        return
    f_ghz = np.asarray(signal_data["freqs_hz"]) * 1e-9
    dt = float(signal_data["dt_s"])
    v_b_fft = np.asarray(signal_data["button_voltage_fft_abs"])
    v_c_fft = np.asarray(signal_data["cable_voltage_fft_abs"])
    v_cable = np.asarray(signal_data["cable_voltage_v"])

    ncols = len(comparison_filters)
    fig, axes = plt.subplots(1, ncols, figsize=(6.6 * ncols, 4.8), squeeze=False)
    for ax, filt in zip(axes[0], comparison_filters):
        vf = apply_analog_filter(v_cable, dt, filt)
        vf_fft = np.abs(np.fft.rfft(vf, n=2 * (len(f_ghz) - 1)))
        h_abs = analog_transfer_abs(np.asarray(signal_data["freqs_hz"]), dt, filt)
        ax2 = ax.twinx()
        ax.loglog(f_ghz[1:], v_b_fft[1:], color="tab:blue", linewidth=1.6, label="Vb")
        ax.loglog(f_ghz[1:], v_c_fft[1:], color="tab:orange", linewidth=1.6, label="Vc")
        ax.loglog(f_ghz[1:], vf_fft[1:], color="tab:red", linewidth=1.6, label=str(filt.get("name", "Vf")))
        ax2.semilogx(f_ghz[1:], h_abs[1:], color="tab:green", linewidth=1.6, label="|H|")
        ax.set_xlabel("f (GHz)")
        ax.set_ylabel("V (arb.)")
        ax2.set_ylabel("|H|")
        ax.set_xlim(1e-2, 1e1)
        ax.set_title(str(filt.get("name", filt.get("type", "filter"))))
        ax.grid(True, which="both", alpha=0.3)
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_fig8_filter_outputs(
    output_path: Path,
    signal_data: dict[str, np.ndarray | float],
    comparison_filters: list[dict[str, Any]],
) -> None:
    if not comparison_filters:
        return
    t_ns = np.asarray(signal_data["t_s"]) * 1e9
    dt = float(signal_data["dt_s"])
    v_cable = np.asarray(signal_data["cable_voltage_v"])

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for filt in comparison_filters:
        vf = apply_analog_filter(v_cable, dt, filt)
        ax.plot(t_ns, vf, linewidth=1.8, label=str(filt.get("name", filt.get("type", "filter"))))
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("Vf (V)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_fig9_button_voltage_cases(
    output_path: Path,
    boundary: Boundary,
    cfg: dict[str, Any],
    signal_cases: list[dict[str, Any]],
) -> None:
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
    cfg_filter = cfg.get("filter", {})
    v_filtered = np.asarray(signal_data["filtered_voltage_v"])
    v_button = np.asarray(signal_data["button_voltage_v"])
    v_cable = np.asarray(signal_data["cable_voltage_v"])
    v_peak = float(np.max(np.abs(v_filtered)))
    v_rms = float(np.sqrt(np.mean(v_filtered**2)))
    cap_pf = float(signal_data["button_capacitance_f"]) * 1e12
    z0 = float(signal_data["characteristic_impedance_ohm"])

    sigma_x_ref_mm = 0.5 * kx_mm * reference_rel_error
    sigma_y_ref_mm = 0.5 * ky_mm * reference_rel_error

    notes = []
    if cfg["chamber"]["kind"].lower() == "polygon":
        notes.append(
            "The sample polygon was taken from the BAR note figures because `BeampositionMonitor.gdf` was not found in the local Research tree."
        )
    if cfg["bunch"]["density"]["kind"].lower() == "gaussian" and float(cfg["bunch"]["density"].get("cutoff_sigma", 0.0)) == 0.0:
        notes.append("The uncapped Gaussian is evaluated on a finite numerical window set by `bunch.longitudinal_grid.no_cut_span_sigma`.")

    lines = [
        "# BAR BPM Analysis Report",
        "",
        f"- Config: `{cfg_path}`",
        f"- Chamber type: `{cfg['chamber']['kind']}`",
        f"- Boundary perimeter: {boundary.perimeter:.3f} mm",
        f"- Button capacitance used in signal model: {cap_pf:.3f} pF",
        f"- Characteristic impedance: {z0:.1f} ohm",
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
    comparison_filters = cfg.get("filter", {}).get("comparison_filters", [])
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
    plot_resolution(resolution_path, rel, sigma_x, sigma_y)
    plot_signal_summary(signal_path, signal_data)
    plot_fig3_impedance_current_spectrum(fig3_path, signal_data)
    plot_fig4_current_voltage(fig4_path, signal_data)
    plot_fig5_cable_io(fig5_path, signal_data)
    plot_fig7_frequency_filters(fig7_path, signal_data, comparison_filters)
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
    parser = argparse.ArgumentParser(description="BAR BPM boundary-element analysis and figure generation.")
    parser.add_argument("config", type=Path, help="Path to the YAML input file.")
    args = parser.parse_args()
    outputs = run(args.config.resolve())
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
