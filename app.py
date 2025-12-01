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


# IEC 60255-151 style constants for IDMT relays.
IEC_CURVES: Dict[str, Tuple[float, float, float]] = {
    "Standard Inverse": (0.14, 0.02, 0.0),
    "Very Inverse": (13.5, 1.0, 0.0),
    "Extremely Inverse": (80.0, 2.0, 0.0),
}


@dataclass
class RelaySettings:
    name: str
    pickup_amps: float
    time_multiplier: float
    curve: str
    instantaneous_pickup: Optional[float]
    instantaneous_time: float


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
class DeviceCurve:
    name: str
    device_type: str
    current_points: np.ndarray
    time_points: np.ndarray
    color: str


DEFAULT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def idmt_trip_times(currents: np.ndarray, settings: RelaySettings) -> np.ndarray:
    """Calculate time dial equation for IEC inverse relay curves.

    The equation is the classic k / ((I/Is)^a - 1) scaled by the time multiplier.
    Instantaneous pickup is applied as a parallel path (min of IDMT time and inst time).
    """
    k, exponent, time_offset = IEC_CURVES[settings.curve]
    multiples = np.maximum(currents / settings.pickup_amps, 1e-9)
    idmt_time = settings.time_multiplier * (k / (np.power(multiples, exponent) - 1.0)) + time_offset

    if settings.instantaneous_pickup is None:
        return idmt_time

    inst_path = np.where(currents >= settings.instantaneous_pickup, settings.instantaneous_time, np.inf)
    return np.minimum(idmt_time, inst_path)


def fuse_trip_times(currents: np.ndarray, settings: FuseSettings) -> np.ndarray:
    """Approximate fuse melting time using a simple I^p model.

    Real-world fuse curves should come from manufacturer data. This placeholder uses
    I^exponent * t = constant to sketch the curve shape.
    """
    times = settings.melting_constant / np.power(np.maximum(currents, 1e-9), settings.exponent)
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
                name=relay.name,
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


def evaluate_coordination(curves: Iterable[DeviceCurve], current_checks: np.ndarray, margin_s: float) -> List[str]:
    """Check that sequential curves maintain the desired time margin.

    The curves are evaluated in the order provided by the user. The check is conservative;
    it flags any point where the next device clears less than `margin_s` slower than the
    downstream device at the same current.
    """
    messages: List[str] = []
    curve_list = list(curves)
    for upstream, downstream in zip(curve_list[:-1], curve_list[1:]):
        upstream_time = np.interp(current_checks, upstream.current_points, upstream.time_points)
        downstream_time = np.interp(current_checks, downstream.current_points, downstream.time_points)
        delta = downstream_time - upstream_time
        min_delta = float(np.nanmin(delta))
        if min_delta < margin_s:
            messages.append(
                f"{downstream.name} coordinates poorly with {upstream.name}: minimum margin {min_delta:.3f}s < {margin_s:.3f}s."
            )
    return messages


def plot_curves(curves: List[DeviceCurve], damage_curves: List[DamageCurve], currents: np.ndarray) -> go.Figure:
    """Render TCC curves on a log-log plot."""
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
            pickup = st.sidebar.number_input("Pickup current (A)", min_value=0.1, value=100.0, key=f"pickup_{idx}")
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
                )
            )
        else:
            pickup = st.sidebar.number_input("Pickup/melting current (A)", min_value=0.1, value=200.0, key=f"pickup_{idx}")
            melting_constant = st.sidebar.number_input(
                "Melting constant (A^p * s)",
                min_value=0.001,
                value=12000.0,
                help="Placeholder constant; use manufacturer data when available.",
                key=f"melting_const_{idx}",
            )
            exponent = st.sidebar.number_input(
                "Exponent p (typ. 2 for thermal)", min_value=1.0, value=2.0, step=0.1, key=f"exp_{idx}"
            )
            minimum_time = st.sidebar.number_input(
                "Minimum clear time (s)", min_value=0.0, value=0.02, step=0.01, key=f"min_time_{idx}"
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

    st.sidebar.markdown("---")
    st.sidebar.subheader("Coordination checks")
    margin_s = st.sidebar.number_input("Required time margin (s)", min_value=0.0, value=0.2, step=0.05)
    current_min = st.sidebar.number_input("Plot min current (A)", min_value=1.0, value=10.0, step=10.0)
    current_max = st.sidebar.number_input("Plot max current (A)", min_value=current_min + 1.0, value=10000.0, step=100.0)

    # Build curve data.
    currents = np.logspace(np.log10(current_min), np.log10(current_max), num=200)
    curves = build_curves(currents, relays, fuses)

    fig = plot_curves(curves, damage_curves, currents)
    st.plotly_chart(fig, use_container_width=True)

    check_currents = np.logspace(np.log10(current_min), np.log10(current_max), num=80)
    messages = evaluate_coordination(curves, check_currents, margin_s)

    if messages:
        st.error("\n".join(messages))
    else:
        st.success("No coordination margin issues detected for the evaluated current range.")

    st.markdown("---")
    st.subheader("How to use")
    st.markdown(
        """
        1. Use the sidebar to add relays and fuses in downstream-to-upstream order.
        2. Enter manufacturer pickup, time-dial, and instantaneous settings for relays.
        3. Replace the placeholder fuse constants with published melting curves when available.
        4. Add optional cable or transformer damage curves to visualize margins.
        5. Adjust the coordination margin to reflect your practice (e.g., 0.2–0.3 s between devices).
        """
    )

    st.caption(
        "This tool does not replace detailed short-circuit studies. Validate settings with manufacturer-provided curves."
    )


if __name__ == "__main__":
    main()
