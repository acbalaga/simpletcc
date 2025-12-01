"""Streamlit app for coordinating TCC curves.

This tool helps compare overcurrent protective devices using standardized IEC and
ANSI/IEEE inverse-time curves plus placeholder fuse/damage representations. It focuses
on clarity and quick visualization rather than manufacturer-accurate curve data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st


@dataclass
class RelaySettings:
    name: str
    pickup_amps: float
    time_multiplier: float
    curve: str
    instantaneous_pickup: Optional[float]
    instantaneous_time: float
    ct_primary: float
    ct_secondary: float
    device_type: str = "Relay"


@dataclass
class FuseSettings:
    name: str
    pickup_amps: float
    melting_constant: float
    exponent: float
    minimum_time: float
    device_type: str = "Fuse"


@dataclass
class DamageCurve:
    name: str
    withstand_constant: float
    exponent: float
    minimum_time: float


@dataclass
class ReferencePoint:
    name: str
    current: float
    time_seconds: float


@dataclass
class DeviceCurve:
    name: str
    device_type: str
    current_points: np.ndarray
    time_points: np.ndarray
    color: str


@dataclass
class CoordinationDiagnostic:
    """Minimum observed margin and current between adjacent protective devices."""

    upstream_device: str
    downstream_device: str
    min_margin_s: float
    current_at_min_a: float
    reference_margin_s: float
    reference_current_a: float


@dataclass
class InverseCurveDefinition:
    k: float
    exponent: float
    time_offset: float
    standard: str
    verified: bool
    note: str


# IEC 60255-151 and IEEE C37.112 style constants for IDMT relays and reclosers.
INVERSE_CURVES: Dict[str, InverseCurveDefinition] = {
    "IEC Standard Inverse": InverseCurveDefinition(
        0.14,
        0.02,
        0.0,
        standard="IEC 60255-151",
        verified=True,
        note="Standardized constants per IEC 60255-151.",
    ),
    "IEC Very Inverse": InverseCurveDefinition(
        13.5,
        1.0,
        0.0,
        standard="IEC 60255-151",
        verified=True,
        note="Standardized constants per IEC 60255-151.",
    ),
    "IEC Extremely Inverse": InverseCurveDefinition(
        80.0,
        2.0,
        0.0,
        standard="IEC 60255-151",
        verified=True,
        note="Standardized constants per IEC 60255-151.",
    ),
    "ANSI Moderately Inverse": InverseCurveDefinition(
        0.0515,
        0.02,
        0.114,
        standard="IEEE C37.112",
        verified=True,
        note="IEEE C37.112 Table 1 (moderately inverse).",
    ),
    "ANSI Very Inverse": InverseCurveDefinition(
        1.0,
        2.0,
        0.0226,
        standard="IEEE C37.112",
        verified=True,
        note="IEEE C37.112 Table 1 (very inverse).",
    ),
    "ANSI Extremely Inverse": InverseCurveDefinition(
        2.0,
        2.0,
        0.00342,
        standard="IEEE C37.112",
        verified=True,
        note="IEEE C37.112 Table 1 (extremely inverse).",
    ),
}

# Approximate fuse presets. These are simplified placeholders based on generic time-current
# shapes (not manufacturer-verified). Replace with catalog data for production studies.
@dataclass
class FusePreset:
    settings: FuseSettings
    verified: bool
    note: str


FUSE_PRESETS: Dict[str, FusePreset] = {
    # Values are illustrative, scaled around the fuse ampere rating. Replace with catalog curves for studies.
    "Class RK5 (time-delay)": FusePreset(
        FuseSettings("Class RK5", 100.0, 90000.0, 2.1, 0.05),
        verified=False,
        note="Placeholder melting curve; replace with manufacturer sheet.",
    ),
    "Class RK1 (current-limiting)": FusePreset(
        FuseSettings("Class RK1", 100.0, 40000.0, 1.7, 0.02),
        verified=False,
        note="Placeholder for current-limiting RK1 profile.",
    ),
    "Class J (fast-acting)": FusePreset(
        FuseSettings("Class J", 100.0, 24000.0, 1.8, 0.015),
        verified=False,
        note="Placeholder fast-acting curve; confirm with catalog data.",
    ),
    "Class T (very fast)": FusePreset(
        FuseSettings("Class T", 100.0, 12000.0, 1.6, 0.01),
        verified=False,
        note="Placeholder for very fast-acting T-class fuse.",
    ),
    "Type K (time-delay)": FusePreset(
        FuseSettings("Type K", 100.0, 80000.0, 2.0, 0.04),
        verified=False,
        note="Placeholder time-delay fuse curve.",
    ),
    "SloFast": FusePreset(
        FuseSettings("SloFast", 100.0, 120000.0, 2.2, 0.06),
        verified=False,
        note="Placeholder dual-element thermal fuse approximation.",
    ),
    "Motor protection": FusePreset(
        FuseSettings("Motor", 100.0, 60000.0, 1.9, 0.03),
        verified=False,
        note="Placeholder motor circuit protector approximation.",
    ),
    "Bussmann Low-Peak LPS-RK (placeholder)": FusePreset(
        FuseSettings("LPS-RK", 100.0, 70000.0, 2.1, 0.05),
        verified=False,
        note="Manufacturer-specific placeholder; swap constants with Bussmann data sheets.",
    ),
    "S&C SMU-20 recloser curve (placeholder)": FusePreset(
        FuseSettings("SMU-20", 100.0, 50000.0, 2.0, 0.04, device_type="Recloser"),
        verified=False,
        note="Recloser melting-style placeholder; replace with published minimum-trip curve.",
    ),
    "Molded-case breaker thermal (placeholder)": FusePreset(
        FuseSettings("MCCB thermal", 100.0, 30000.0, 2.5, 0.02, device_type="Molded-case breaker"),
        verified=False,
        note="Thermal-magnetic placeholder; use manufacturer time-current data for studies.",
    ),
}

# Approximate withstand curves for common equipment. Constants are illustrative only and
# should be replaced with utility or manufacturer data before field use.
DEFAULT_DAMAGE_CURVES: List[DamageCurve] = [
    DamageCurve("LV Cu cable 90C (approx)", withstand_constant=45000.0, exponent=2.0, minimum_time=0.1),
    DamageCurve("LV Al cable 75C (approx)", withstand_constant=30000.0, exponent=2.0, minimum_time=0.1),
    DamageCurve("Dry-type transformer (IEEE C57.109 approx)", withstand_constant=120000.0, exponent=1.0, minimum_time=1.0),
]


DEFAULT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
DEVICE_TYPE_OPTIONS = ["Relay", "Recloser", "Molded-case breaker", "Fuse"]


def idmt_trip_times(currents: np.ndarray, settings: RelaySettings) -> np.ndarray:
    """Calculate time dial equation for IEC inverse relay curves.

    The equation is the classic k / ((I/Is)^a - 1) scaled by the time multiplier.
    Instantaneous pickup is applied as a parallel path (min of IDMT time and inst time).
    """
    curve = INVERSE_CURVES[settings.curve]
    k, exponent, time_offset = curve.k, curve.exponent, curve.time_offset
    ct_ratio = settings.ct_primary / max(settings.ct_secondary, 1e-9)
    pickup_primary = settings.pickup_amps * ct_ratio
    multiples = np.maximum(currents / pickup_primary, 1e-9)

    # The IEC equation is only defined for multiples > 1. Clamp lower values to NaN
    # so they do not render on the log-log plot and avoid negative times.
    valid_mask = multiples > 1.0
    idmt_time = np.full_like(currents, np.nan)
    idmt_time[valid_mask] = (
        settings.time_multiplier * (k / (np.power(multiples[valid_mask], exponent) - 1.0)) + time_offset
    )

    if settings.instantaneous_pickup is None:
        return idmt_time

    inst_pickup_primary = settings.instantaneous_pickup * ct_ratio
    inst_path = np.where(currents >= inst_pickup_primary, settings.instantaneous_time, np.inf)
    return np.minimum(idmt_time, inst_path)


def fuse_trip_times(currents: np.ndarray, settings: FuseSettings) -> np.ndarray:
    """Approximate fuse melting time using a simple I^p model.

    Real-world fuse curves should come from manufacturer data. This placeholder uses
    I^exponent * t = constant to sketch the curve shape.
    """
    multiples = np.maximum(currents / max(settings.pickup_amps, 1e-9), 1e-9)
    times = settings.melting_constant / np.power(multiples, settings.exponent)
    times = np.maximum(times, settings.minimum_time)
    return times


def damage_withstand_times(currents: np.ndarray, curve: DamageCurve) -> np.ndarray:
    """Calculate an I^p withstand (damage) curve for cables or transformers."""
    times = curve.withstand_constant / np.power(np.maximum(currents, 1e-9), curve.exponent)
    return np.maximum(times, curve.minimum_time)


def build_curves(currents: np.ndarray, devices: List[RelaySettings | FuseSettings]) -> List[DeviceCurve]:
    """Combine device definitions into plottable curves preserving user ordering."""

    curves: List[DeviceCurve] = []
    for idx, device in enumerate(devices):
        if isinstance(device, RelaySettings):
            times = idmt_trip_times(currents, device)
            name = f"{device.name} (CT {device.ct_primary:.0f}/{device.ct_secondary:.1f})"
            device_type = device.device_type
        elif isinstance(device, FuseSettings):
            times = fuse_trip_times(currents, device)
            name = device.name
            device_type = device.device_type
        else:
            # This should not happen with current UI controls but keeps the function safe if extended.
            raise TypeError("Unsupported device type for curve generation")

        curves.append(
            DeviceCurve(
                name=name,
                device_type=device_type,
                current_points=currents,
                time_points=times,
                color=DEFAULT_COLORS[idx % len(DEFAULT_COLORS)],
            )
        )
    return curves


def evaluate_coordination(
    curves: Iterable[DeviceCurve],
    current_checks: np.ndarray,
    margin_s: float,
    return_diagnostics: bool = False,
    reference_current: Optional[float] = None,
) -> Tuple[List[str], List[CoordinationDiagnostic]]:
    """Check sequential curves for time margin and optionally return diagnostics.

    The curves are evaluated in the order provided by the user and treated as downstream
    to upstream. The check is conservative; it flags any point where the upstream device
    clears less than `margin_s` slower than the downstream device at the same current.
    When ``return_diagnostics`` is True the function also returns the minimum observed
    margin and the current where it occurs for each adjacent pair, enabling the UI to
    show actionable detail. Diagnostics are returned as an empty list when the flag is
    False. When ``reference_current`` is supplied, the function also reports the margin
    at that current (if it lies within both curves' domains).
    """
    messages: List[str] = []
    diagnostics: List[CoordinationDiagnostic] = []
    curve_list = list(curves)
    for pair_index, (downstream_curve, upstream_curve) in enumerate(
        zip(curve_list[:-1], curve_list[1:])
    ):
        # Only compare within the portion of each curve that has finite time values to avoid
        # interpolating through undefined regions (e.g., below pickup).
        downstream_mask = np.isfinite(downstream_curve.time_points) & np.isfinite(
            downstream_curve.current_points
        )
        upstream_mask = np.isfinite(upstream_curve.time_points) & np.isfinite(
            upstream_curve.current_points
        )

        downstream_currents = downstream_curve.current_points[downstream_mask]
        downstream_times = downstream_curve.time_points[downstream_mask]
        upstream_currents = upstream_curve.current_points[upstream_mask]
        upstream_times = upstream_curve.time_points[upstream_mask]

        if downstream_currents.size == 0 or upstream_currents.size == 0:
            diagnostics.append(
                CoordinationDiagnostic(
                    upstream_device=upstream_curve.name,
                    downstream_device=downstream_curve.name,
                    min_margin_s=float("nan"),
                    current_at_min_a=float("nan"),
                    reference_margin_s=float("nan"),
                    reference_current_a=float(reference_current)
                    if reference_current is not None
                    else float("nan"),
                )
            )
            continue

        overlap_min = max(float(np.min(downstream_currents)), float(np.min(upstream_currents)))
        overlap_max = min(float(np.max(downstream_currents)), float(np.max(upstream_currents)))
        overlap_checks = current_checks[(current_checks >= overlap_min) & (current_checks <= overlap_max)]

        if overlap_min >= overlap_max or overlap_checks.size == 0:
            diagnostics.append(
                CoordinationDiagnostic(
                    upstream_device=upstream_curve.name,
                    downstream_device=downstream_curve.name,
                    min_margin_s=float("nan"),
                    current_at_min_a=float("nan"),
                    reference_margin_s=float("nan"),
                    reference_current_a=float(reference_current)
                    if reference_current is not None
                    else float("nan"),
                )
            )
            continue

        downstream_time = np.interp(overlap_checks, downstream_currents, downstream_times)
        upstream_time = np.interp(overlap_checks, upstream_currents, upstream_times)
        delta = upstream_time - downstream_time

        finite_mask = np.isfinite(delta)
        if not np.any(finite_mask):
            min_delta = float("nan")
            current_at_min = float("nan")
        else:
            finite_delta = delta[finite_mask]
            finite_currents = overlap_checks[finite_mask]
            min_idx = int(np.argmin(finite_delta))
            min_delta = float(finite_delta[min_idx])
            current_at_min = float(finite_currents[min_idx])

        reference_margin = float("nan")
        if (
            reference_current is not None
            and overlap_min <= reference_current <= overlap_max
        ):
            downstream_at_ref = float(
                np.interp(reference_current, downstream_currents, downstream_times)
            )
            upstream_at_ref = float(
                np.interp(reference_current, upstream_currents, upstream_times)
            )
            reference_margin = upstream_at_ref - downstream_at_ref

        diagnostics.append(
            CoordinationDiagnostic(
                upstream_device=upstream_curve.name,
                downstream_device=downstream_curve.name,
                min_margin_s=min_delta,
                current_at_min_a=current_at_min,
                reference_margin_s=reference_margin,
                reference_current_a=float(reference_current)
                if reference_current is not None
                else float("nan"),
            )
        )

        if np.isfinite(min_delta) and min_delta < margin_s:
            messages.append(
                (
                    f"Device {pair_index + 1} (downstream: {downstream_curve.name}) and "
                    f"Device {pair_index + 2} (upstream: {upstream_curve.name}) coordinate poorly: "
                    f"minimum margin {min_delta:.3f}s < {margin_s:.3f}s."
                )
            )

        if np.isfinite(reference_margin) and reference_margin < margin_s:
            messages.append(
                (
                    f"At {reference_current:.2f} A, Device {pair_index + 1} (downstream: {downstream_curve.name}) "
                    f"and Device {pair_index + 2} (upstream: {upstream_curve.name}) have only {reference_margin:.3f}s margin "
                    f"(< {margin_s:.3f}s)."
                )
            )

    if return_diagnostics:
        return messages, diagnostics

    return messages, []


def plot_curves(
    curves: List[DeviceCurve],
    damage_curves: List[DamageCurve],
    currents: np.ndarray,
    reference_points: Optional[List[ReferencePoint]] = None,
    coordination_current: Optional[float] = None,
) -> go.Figure:
    """Render TCC curves on a log-log plot with optional reference markers."""
    fig = go.Figure()
    for curve in curves:
        fig.add_trace(
            go.Scatter(
                x=curve.current_points,
                y=curve.time_points,
                mode="lines",
                name=f"{curve.device_type}: {curve.name}",
                line=dict(color=curve.color, width=3),
            )
        )

    for curve in damage_curves:
        fig.add_trace(
            go.Scatter(
                x=currents,
                y=damage_withstand_times(currents, curve),
                mode="lines",
                name=f"Damage: {curve.name}",
                line=dict(color="#444", dash="dash"),
            )
        )

    if reference_points:
        for point in reference_points:
            fig.add_trace(
                go.Scatter(
                    x=[point.current],
                    y=[point.time_seconds],
                    mode="markers+text",
                    name=point.name,
                    marker=dict(color="#111", size=10, symbol="x"),
                    text=[point.name],
                    textposition="top center",
                    hovertemplate="%{text}<br>Current: %{x:.2f} A<br>Time: %{y:.3f} s<extra></extra>",
                )
            )

    if coordination_current:
        times_for_bounds: List[float] = []
        for curve in curves:
            times_for_bounds.extend(np.asarray(curve.time_points, dtype=float))
        for damage_curve in damage_curves:
            times_for_bounds.extend(damage_withstand_times(currents, damage_curve))
        if reference_points:
            times_for_bounds.extend(point.time_seconds for point in reference_points)

        finite_times = np.asarray(times_for_bounds, dtype=float)
        finite_times = finite_times[np.isfinite(finite_times) & (finite_times > 0)]
        y_min = float(np.nanmin(finite_times)) if finite_times.size > 0 else 0.001
        y_max = float(np.nanmax(finite_times)) if finite_times.size > 0 else 10.0
        # Ensure the line spans a visible portion of the log axis.
        if y_min == y_max:
            y_min *= 0.8
            y_max *= 1.2

        fig.add_trace(
            go.Scatter(
                x=[coordination_current, coordination_current],
                y=[y_min, y_max],
                mode="lines",
                name="Coordination current",
                line=dict(color="#111", dash="dot", width=2),
                hovertemplate="Coordination current<br>Current: %{x:.2f} A<extra></extra>",
            )
        )

    fig.update_xaxes(title="Current (A)", type="log", exponentformat="power")
    fig.update_yaxes(title="Time (s)", type="log", exponentformat="power")
    fig.update_layout(height=600, legend=dict(orientation="h"))
    return fig


def sidebar_device_inputs(device_count: int) -> List[RelaySettings | FuseSettings]:
    """Collect user-defined relay and fuse settings from the sidebar."""

    devices: List[RelaySettings | FuseSettings] = []

    for idx in range(device_count):
        st.sidebar.markdown("---")
        st.sidebar.subheader(f"Device {idx + 1}")
        device_type = st.sidebar.selectbox(
            "Device type",
            options=DEVICE_TYPE_OPTIONS,
            key=f"type_{idx}",
            help=(
                "Relays/reclosers use inverse-time curves. Molded-case breakers and fuses"
                " rely on melting-style placeholders; replace constants with catalog data."
            ),
        )
        name = st.sidebar.text_input("Name", value=f"Device {idx + 1}", key=f"name_{idx}")

        if device_type in {"Relay", "Recloser"}:
            curve_options = list(INVERSE_CURVES.keys())
            curve_labels = {
                key: f"{key} ({'verified' if val.verified else 'placeholder'} - {val.standard})"
                for key, val in INVERSE_CURVES.items()
            }
            curve = st.sidebar.selectbox(
                "Curve family",
                curve_options,
                key=f"curve_{idx}",
                format_func=lambda key: curve_labels.get(key, key),
                help="IEC curves are standardized; ANSI MI/VI/EI curves follow IEEE C37.112.",
            )
            st.sidebar.caption(
                f"{curve_labels[curve]}. Notes: {INVERSE_CURVES[curve].note}"
            )
            ct_primary = st.sidebar.number_input(
                "CT primary (A)", min_value=1.0, value=400.0, step=5.0, key=f"ct_primary_{idx}"
            )
            ct_secondary = st.sidebar.number_input(
                "CT secondary (A)", min_value=0.1, value=5.0, step=0.1, key=f"ct_secondary_{idx}"
            )
            pickup = st.sidebar.number_input(
                "Pickup current setting (A secondary)",
                min_value=0.1,
                value=5.0,
                help="Secondary pickup setting; primary pickup uses CT ratio.",
                key=f"pickup_{idx}",
            )
            tms = st.sidebar.number_input("Time multiplier (TMS)", min_value=0.01, value=0.1, key=f"tms_{idx}")
            inst_pickup = st.sidebar.number_input(
                "Instantaneous pickup (A)",
                min_value=0.0,
                value=0.0,
                help="Set to 0 to disable the instantaneous element.",
                key=f"inst_{idx}",
            )
            inst_time = st.sidebar.number_input(
                "Instantaneous clearing time (s)", min_value=0.01, value=0.05, key=f"inst_time_{idx}"
            )
            devices.append(
                RelaySettings(
                    name=name,
                    pickup_amps=pickup,
                    time_multiplier=tms,
                    curve=curve,
                    instantaneous_pickup=inst_pickup if inst_pickup > 0 else None,
                    instantaneous_time=inst_time,
                    ct_primary=ct_primary,
                    ct_secondary=ct_secondary,
                    device_type=device_type,
                )
            )
        else:
            preset_options = ["Custom"] + [
                name
                for name, preset in FUSE_PRESETS.items()
                if preset.settings.device_type == device_type or preset.settings.device_type == "Fuse"
            ]
            preset_label = st.sidebar.selectbox(
                "Preset fuse type",
                options=preset_options,
                index=0,
                key=f"fuse_preset_{idx}",
                format_func=lambda key: (
                    key
                    if key == "Custom"
                    else f"{key} ({'verified' if FUSE_PRESETS[key].verified else 'placeholder'})"
                ),
                help="Preset constants are illustrative; manufacturer-specific entries are marked as placeholders unless noted.",
            )
            preset = FUSE_PRESETS.get(preset_label)
            preset_settings = preset.settings if preset else None
            if preset:
                st.sidebar.caption(preset.note)

            # Update session defaults when a preset changes so the inputs reflect the selected curve.
            preset_state_key = f"fuse_last_preset_{idx}"
            if preset_label != "Custom" and st.session_state.get(preset_state_key) != preset_label:
                st.session_state[preset_state_key] = preset_label
                st.session_state[f"pickup_{idx}"] = preset_settings.pickup_amps
                st.session_state[f"melting_const_{idx}"] = preset_settings.melting_constant
                st.session_state[f"exp_{idx}"] = preset_settings.exponent
                st.session_state[f"min_time_{idx}"] = preset_settings.minimum_time

            pickup_default = (
                preset_settings.pickup_amps
                if preset_settings
                else st.session_state.get(f"pickup_{idx}", 200.0)
            )
            melting_default = (
                preset_settings.melting_constant
                if preset_settings
                else st.session_state.get(f"melting_const_{idx}", 12000.0)
            )
            exponent_default = (
                preset_settings.exponent
                if preset_settings
                else st.session_state.get(f"exp_{idx}", 2.0)
            )
            min_time_default = (
                preset_settings.minimum_time
                if preset_settings
                else st.session_state.get(f"min_time_{idx}", 0.02)
            )

            pickup = st.sidebar.number_input(
                "Pickup/melting current (A)", min_value=0.1, value=pickup_default, key=f"pickup_{idx}"
            )
            melting_constant = st.sidebar.number_input(
                "Melting constant (A^p * s)",
                min_value=0.001,
                value=melting_default,
                help="Placeholder constant; use manufacturer data when available.",
                key=f"melting_const_{idx}",
            )
            exponent = st.sidebar.number_input(
                "Exponent p (typ. 2 for thermal)", min_value=1.0, value=exponent_default, step=0.1, key=f"exp_{idx}"
            )
            minimum_time = st.sidebar.number_input(
                "Minimum clear time (s)", min_value=0.0, value=min_time_default, step=0.01, key=f"min_time_{idx}"
            )
            st.sidebar.caption(
                "Placeholders are intended for quick studies only. Replace with manufacturer datasets before issuing settings."
            )
            devices.append(
                FuseSettings(
                    name=name,
                    pickup_amps=pickup,
                    melting_constant=melting_constant,
                    exponent=exponent,
                    minimum_time=minimum_time,
                    device_type=preset_settings.device_type if preset_settings else device_type,
                )
            )

    return devices


def sidebar_damage_inputs() -> List[DamageCurve]:
    """Optional cable/transformer damage curve inputs."""
    damage_curves: List[DamageCurve] = []
    st.sidebar.markdown("---")
    st.sidebar.subheader("Damage curves (optional)")
    preset_names = [curve.name for curve in DEFAULT_DAMAGE_CURVES]
    selected_presets = st.sidebar.multiselect(
        "Add preset damage curves",
        options=preset_names,
        default=[],
        help="Preset withstand curves are approximate; replace with utility/manufacturer data when available.",
    )
    for name in selected_presets:
        preset_curve = next(curve for curve in DEFAULT_DAMAGE_CURVES if curve.name == name)
        key_prefix = f"damage_preset_{name.replace(' ', '_')[:30]}"
        constant_default = st.session_state.get(f"{key_prefix}_constant", preset_curve.withstand_constant)
        exponent_default = st.session_state.get(f"{key_prefix}_exponent", preset_curve.exponent)
        min_time_default = st.session_state.get(f"{key_prefix}_min_time", preset_curve.minimum_time)

        with st.sidebar.expander(f"{name} settings", expanded=False):
            constant = st.number_input(
                "Withstand constant (A^p * s)",
                min_value=0.001,
                value=constant_default,
                help="Edit the preset to reflect updated thermal limits.",
                key=f"{key_prefix}_constant",
            )
            exponent = st.number_input(
                "Exponent p",
                min_value=1.0,
                value=exponent_default,
                step=0.1,
                key=f"{key_prefix}_exponent",
            )
            minimum_time = st.number_input(
                "Minimum withstand time (s)",
                min_value=0.0,
                value=min_time_default,
                step=0.01,
                key=f"{key_prefix}_min_time",
            )

        damage_curves.append(
            DamageCurve(name=preset_curve.name, withstand_constant=constant, exponent=exponent, minimum_time=minimum_time)
        )
    damage_count = st.sidebar.number_input("Number of damage curves", min_value=0, value=0, step=1)
    for idx in range(int(damage_count)):
        name = st.sidebar.text_input("Name", value=f"Damage {idx + 1}", key=f"damage_name_{idx}")
        constant = st.sidebar.number_input(
            "Withstand constant (A^p * s)",
            min_value=0.001,
            value=20000.0,
            help="Use utility or equipment data when available.",
            key=f"damage_const_{idx}",
        )
        exponent = st.sidebar.number_input(
            "Exponent p", min_value=1.0, value=1.0, step=0.1, key=f"damage_exp_{idx}"
        )
        minimum_time = st.sidebar.number_input(
            "Minimum withstand time (s)", min_value=0.0, value=0.1, step=0.01, key=f"damage_min_{idx}"
        )
        damage_curves.append(
            DamageCurve(name=name, withstand_constant=constant, exponent=exponent, minimum_time=minimum_time)
        )
    return damage_curves


def sidebar_reference_points() -> Tuple[List[ReferencePoint], Optional[float]]:
    """Optional point markers (e.g., full-load current) to overlay on the plot."""
    points: List[ReferencePoint] = []
    st.sidebar.markdown("---")
    st.sidebar.subheader("Reference point")
    add_point = st.sidebar.checkbox("Plot a reference point", value=False)
    if add_point:
        name = st.sidebar.text_input("Label", value="Full-load current", key="ref_name")
        current = st.sidebar.number_input(
            "Current (A)",
            min_value=0.1,
            value=500.0,
            help="Plot a marker at this current. Use primary amps for consistency with device curves.",
            key="ref_current",
        )
        time_seconds = st.sidebar.number_input(
            "Time (s)",
            min_value=0.001,
            value=10.0,
            help="Place the point at this time value (e.g., 1.0 s for full-load).",
            key="ref_time",
        )
        points.append(ReferencePoint(name=name, current=current, time_seconds=time_seconds))

    coordination_current: Optional[float] = None
    st.sidebar.markdown("---")
    st.sidebar.subheader("Coordination current")
    add_coordination_line = st.sidebar.checkbox("Show coordination current line", value=False)
    if add_coordination_line:
        coordination_current = st.sidebar.number_input(
            "Coordination current (A)",
            min_value=0.1,
            value=500.0,
            help="Draw a vertical line at this current to visualize coordination margins.",
            key="coordination_current",
        )

    return points, coordination_current


def main() -> None:
    st.set_page_config(page_title="TCC Coordinator", layout="wide")
    st.title("Time-Current Characteristic (TCC) Coordinator")
    st.write(
        """
        Configure overcurrent devices and visualize their time-current curves. The app offers IEC 60255 and
        ANSI/IEEE MI/VI/EI inverse-time families for relays or reclosers plus simple I^p placeholders for fuses,
        molded-case breakers, and damage curves. Replace placeholder constants with manufacturer data for production
        studies.
        """
    )

    st.sidebar.header("Configuration")
    device_count = st.sidebar.number_input("Number of devices", min_value=1, value=2, step=1)
    devices = sidebar_device_inputs(int(device_count))
    damage_curves = sidebar_damage_inputs()
    reference_points, coordination_current = sidebar_reference_points()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Coordination checks")
    margin_s = st.sidebar.number_input("Required time margin (s)", min_value=0.0, value=0.2, step=0.05)
    current_min = st.sidebar.number_input("Plot min current (A)", min_value=1.0, value=10.0, step=10.0)
    current_max = st.sidebar.number_input("Plot max current (A)", min_value=current_min + 1.0, value=10000.0, step=100.0)

    # Build curve data.
    currents = np.logspace(np.log10(current_min), np.log10(current_max), num=200)
    curves = build_curves(currents, devices)

    fig = plot_curves(
        curves,
        damage_curves,
        currents,
        reference_points=reference_points,
        coordination_current=coordination_current,
    )
    st.plotly_chart(fig, use_container_width=True)

    check_currents = np.logspace(np.log10(current_min), np.log10(current_max), num=80)
    messages, diagnostics = evaluate_coordination(
        curves,
        check_currents,
        margin_s,
        return_diagnostics=True,
        reference_current=coordination_current,
    )

    if diagnostics:
        st.subheader("Coordination summary")
        st.table(
            [
                {
                    "Downstream device": f"Device {idx + 1}: {diag.downstream_device}",
                    "Upstream device": f"Device {idx + 2}: {diag.upstream_device}",
                    "Min margin (s)": round(diag.min_margin_s, 4)
                    if np.isfinite(diag.min_margin_s)
                    else "N/A",
                    "Current at min (A)": round(diag.current_at_min_a, 2)
                    if np.isfinite(diag.current_at_min_a)
                    else "N/A",
                    "Margin at coord. current (s)": round(diag.reference_margin_s, 4)
                    if np.isfinite(diag.reference_margin_s)
                    else "N/A",
                }
                for idx, diag in enumerate(diagnostics)
            ]
        )

    if messages:
        st.error("\n".join(messages))
    else:
        st.success("No coordination margin issues detected for the evaluated current range.")

    st.markdown("---")
    st.subheader("How to use")
    st.markdown(
        """
        1. Use the sidebar to add relays, reclosers, molded-case breakers, and fuses in downstream-to-upstream order.
        2. Enter manufacturer pickup, time-dial, and instantaneous settings for relays, along with CT ratios. Choose IEC or
           ANSI MI/VI/EI curve families as needed.
        3. Pick a preset fuse/breaker family or enter manufacturer melting constants and exponents. Sidebar labels call out
           placeholders versus standardized/verified curves.
        4. Add optional cable or transformer damage curves (preset or custom) to visualize margins.
        5. Overlay a reference marker (e.g., full-load current) to check operating points against curves.
        6. Adjust the coordination margin to reflect your practice (e.g., 0.2–0.3 s between devices).
        """
    )

    st.caption(
        "This tool does not replace detailed short-circuit studies. Preset fuse/damage data are illustrative; validate all settings with manufacturer or utility-provided curves."
    )


if __name__ == "__main__":
    main()
