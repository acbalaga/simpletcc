"""Streamlit app for coordinating TCC curves.

This tool helps compare overcurrent protective devices using simplified
IEC relay curves and placeholder fuse/damage representations. It focuses on clarity
and quick visualization rather than manufacturer-accurate curve data.
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


@dataclass
class FuseSettings:
    name: str
    pickup_amps: float
    melting_constant: float
    exponent: float
    minimum_time: float


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


# IEC 60255-151 style constants for IDMT relays.
IEC_CURVES: Dict[str, Tuple[float, float, float]] = {
    "Standard Inverse": (0.14, 0.02, 0.0),
    "Very Inverse": (13.5, 1.0, 0.0),
    "Extremely Inverse": (80.0, 2.0, 0.0),
}

# Approximate fuse presets. These are simplified placeholders based on generic time-current
# shapes (not manufacturer-verified). Replace with catalog data for production studies.
FUSE_PRESETS: Dict[str, FuseSettings] = {
    # Values are illustrative, scaled around the fuse ampere rating. Replace with catalog curves for studies.
    "Class RK5 (time-delay)": FuseSettings("Class RK5", 100.0, 90000.0, 2.1, 0.05),
    "Class RK1 (current-limiting)": FuseSettings("Class RK1", 100.0, 40000.0, 1.7, 0.02),
    "Class J (fast-acting)": FuseSettings("Class J", 100.0, 24000.0, 1.8, 0.015),
    "Class T (very fast)": FuseSettings("Class T", 100.0, 12000.0, 1.6, 0.01),
    "Type K (time-delay)": FuseSettings("Type K", 100.0, 80000.0, 2.0, 0.04),
    "SloFast": FuseSettings("SloFast", 100.0, 120000.0, 2.2, 0.06),
    "Motor protection": FuseSettings("Motor", 100.0, 60000.0, 1.9, 0.03),
}

# Approximate withstand curves for common equipment. Constants are illustrative only and
# should be replaced with utility or manufacturer data before field use.
DEFAULT_DAMAGE_CURVES: List[DamageCurve] = [
    DamageCurve("LV Cu cable 90C (approx)", withstand_constant=45000.0, exponent=2.0, minimum_time=0.1),
    DamageCurve("LV Al cable 75C (approx)", withstand_constant=30000.0, exponent=2.0, minimum_time=0.1),
    DamageCurve("Dry-type transformer (IEEE C57.109 approx)", withstand_constant=120000.0, exponent=1.0, minimum_time=1.0),
]


DEFAULT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def idmt_trip_times(currents: np.ndarray, settings: RelaySettings) -> np.ndarray:
    """Calculate time dial equation for IEC inverse relay curves.

    The equation is the classic k / ((I/Is)^a - 1) scaled by the time multiplier.
    Instantaneous pickup is applied as a parallel path (min of IDMT time and inst time).
    """
    k, exponent, time_offset = IEC_CURVES[settings.curve]
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


def build_curves(currents: np.ndarray, relays: List[RelaySettings], fuses: List[FuseSettings]) -> List[DeviceCurve]:
    """Combine relay and fuse definitions into plottable curves."""
    curves: List[DeviceCurve] = []
    for idx, relay in enumerate(relays):
        times = idmt_trip_times(currents, relay)
        curves.append(
            DeviceCurve(
                name=f"{relay.name} (CT {relay.ct_primary:.0f}/{relay.ct_secondary:.1f})",
                device_type="Relay",
                current_points=currents,
                time_points=times,
                color=DEFAULT_COLORS[idx % len(DEFAULT_COLORS)],
            )
        )

    start_idx = len(relays)
    for idx, fuse in enumerate(fuses):
        times = fuse_trip_times(currents, fuse)
        curves.append(
            DeviceCurve(
                name=fuse.name,
                device_type="Fuse",
                current_points=currents,
                time_points=times,
                color=DEFAULT_COLORS[(start_idx + idx) % len(DEFAULT_COLORS)],
            )
        )
    return curves


def evaluate_coordination(
    curves: Iterable[DeviceCurve],
    current_checks: np.ndarray,
    margin_s: float,
    return_diagnostics: bool = False,
) -> Tuple[List[str], List[CoordinationDiagnostic]]:
    """Check sequential curves for time margin and optionally return diagnostics.

    The curves are evaluated in the order provided by the user and treated as downstream
    to upstream. The check is conservative; it flags any point where the upstream device
    clears less than `margin_s` slower than the downstream device at the same current.
    When ``return_diagnostics`` is True the function also returns the minimum observed
    margin and the current where it occurs for each adjacent pair, enabling the UI to
    show actionable detail. Diagnostics are returned as an empty list when the flag is
    False.
    """
    messages: List[str] = []
    diagnostics: List[CoordinationDiagnostic] = []
    curve_list = list(curves)
    for downstream_curve, upstream_curve in zip(curve_list[:-1], curve_list[1:]):
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

        diagnostics.append(
            CoordinationDiagnostic(
                upstream_device=upstream_curve.name,
                downstream_device=downstream_curve.name,
                min_margin_s=min_delta,
                current_at_min_a=current_at_min,
            )
        )

        if np.isfinite(min_delta) and min_delta < margin_s:
            messages.append(
                f"Upstream device {upstream_curve.name} coordinates poorly with downstream device {downstream_curve.name}: minimum margin {min_delta:.3f}s < {margin_s:.3f}s."
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


def sidebar_device_inputs(device_count: int) -> Tuple[List[RelaySettings], List[FuseSettings]]:
    """Collect user-defined relay and fuse settings from the sidebar."""
    relays: List[RelaySettings] = []
    fuses: List[FuseSettings] = []

    for idx in range(device_count):
        st.sidebar.markdown("---")
        st.sidebar.subheader(f"Device {idx + 1}")
        device_type = st.sidebar.selectbox(
            "Device type",
            options=["Relay", "Fuse"],
            key=f"type_{idx}",
        )
        name = st.sidebar.text_input("Name", value=f"Device {idx + 1}", key=f"name_{idx}")

        if device_type == "Relay":
            curve = st.sidebar.selectbox("Curve family", list(IEC_CURVES.keys()), key=f"curve_{idx}")
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
            relays.append(
                RelaySettings(
                    name=name,
                    pickup_amps=pickup,
                    time_multiplier=tms,
                    curve=curve,
                    instantaneous_pickup=inst_pickup if inst_pickup > 0 else None,
                    instantaneous_time=inst_time,
                    ct_primary=ct_primary,
                    ct_secondary=ct_secondary,
                )
            )
        else:
            preset_label = st.sidebar.selectbox(
                "Preset fuse type",
                options=["Custom"] + list(FUSE_PRESETS.keys()),
                index=0,
                key=f"fuse_preset_{idx}",
                help="Preset constants are illustrative; replace with manufacturer data for studies.",
            )
            preset = FUSE_PRESETS.get(preset_label)

            # Update session defaults when a preset changes so the inputs reflect the selected curve.
            preset_state_key = f"fuse_last_preset_{idx}"
            if preset_label != "Custom" and st.session_state.get(preset_state_key) != preset_label:
                st.session_state[preset_state_key] = preset_label
                st.session_state[f"pickup_{idx}"] = preset.pickup_amps
                st.session_state[f"melting_const_{idx}"] = preset.melting_constant
                st.session_state[f"exp_{idx}"] = preset.exponent
                st.session_state[f"min_time_{idx}"] = preset.minimum_time

            pickup_default = preset.pickup_amps if preset else st.session_state.get(f"pickup_{idx}", 200.0)
            melting_default = preset.melting_constant if preset else st.session_state.get(f"melting_const_{idx}", 12000.0)
            exponent_default = preset.exponent if preset else st.session_state.get(f"exp_{idx}", 2.0)
            min_time_default = preset.minimum_time if preset else st.session_state.get(f"min_time_{idx}", 0.02)

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
            fuses.append(
                FuseSettings(
                    name=name,
                    pickup_amps=pickup,
                    melting_constant=melting_constant,
                    exponent=exponent,
                    minimum_time=minimum_time,
                )
            )

    return relays, fuses


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
        Configure overcurrent devices and visualize their time-current curves. The app uses
        IEC-style IDMT equations for relays and simple I^p placeholders for fuses and damage curves.
        Replace placeholder constants with manufacturer data for production studies.
        """
    )

    st.sidebar.header("Configuration")
    device_count = st.sidebar.number_input("Number of devices", min_value=1, value=2, step=1)
    relays, fuses = sidebar_device_inputs(int(device_count))
    damage_curves = sidebar_damage_inputs()
    reference_points, coordination_current = sidebar_reference_points()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Coordination checks")
    margin_s = st.sidebar.number_input("Required time margin (s)", min_value=0.0, value=0.2, step=0.05)
    current_min = st.sidebar.number_input("Plot min current (A)", min_value=1.0, value=10.0, step=10.0)
    current_max = st.sidebar.number_input("Plot max current (A)", min_value=current_min + 1.0, value=10000.0, step=100.0)

    # Build curve data.
    currents = np.logspace(np.log10(current_min), np.log10(current_max), num=200)
    curves = build_curves(currents, relays, fuses)

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
        curves, check_currents, margin_s, return_diagnostics=True
    )

    if diagnostics:
        st.subheader("Coordination summary")
        st.table(
            [
                {
                    "Upstream device": diag.upstream_device,
                    "Downstream device": diag.downstream_device,
                    "Min margin (s)": round(diag.min_margin_s, 4)
                    if np.isfinite(diag.min_margin_s)
                    else "N/A",
                    "Current at min (A)": round(diag.current_at_min_a, 2)
                    if np.isfinite(diag.current_at_min_a)
                    else "N/A",
                }
                for diag in diagnostics
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
        1. Use the sidebar to add relays and fuses in downstream-to-upstream order.
        2. Enter manufacturer pickup, time-dial, and instantaneous settings for relays, along with CT ratios.
        3. Pick a preset fuse family or enter manufacturer melting constants and exponents.
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
