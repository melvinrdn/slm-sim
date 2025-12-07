# slm-sim

A tool to simulate laser beam propagation after applying a phase mask on a Spatial Light Modulator (SLM).

Features:
- SLM phase application
- Fourier propagation to the focus  
- Basic Z‑scan
- Simple plotting utilities

## Usage

Run an example:

```bash
python examples/vortex_beam.py
```

Outputs are saved in `fig/`.

## Structure

```
examples/       Example scripts
src/slm_sim/    Core simulation + utilities
fig/            Generated figures
```

## Install

```
pip install -e .
```

MIT License.
