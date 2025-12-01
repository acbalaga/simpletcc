# TCC Coordinator (Streamlit)

A minimal Streamlit app for coordinating 50/51 overcurrent protective devices.
It visualizes relay, fuse, and optional damage curves so you can validate time
margins between downstream and upstream devices.

## Features
- IEC inverse-time relay curves with configurable pickup, time multiplier, and instantaneous elements.
- Placeholder fuse melting curves using an I^p model (replace constants with manufacturer data).
- Optional cable/transformer damage curves for margin checks.
- Log-log Plotly chart with coordination warnings based on a user-defined margin.

## Running locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open the provided local URL and configure devices in the sidebar (downstream to upstream order).

## Notes
- Curve equations are simplified for quick studies. Always verify settings against manufacturer
  TCC data and detailed short-circuit analysis before applying them in the field.
- The fuse and damage curves rely on placeholder constants; adjust them per equipment data.

## Suggested quick test
- Run `python -m compileall app.py` to ensure the app imports cleanly after edits.
