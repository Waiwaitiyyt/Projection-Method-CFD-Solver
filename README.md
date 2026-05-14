# Projection Method CFD Solver

A 2D computational fluid dynamics (CFD) simulator written in Python.

## Overview

`sim.py` implements a finite-difference Navier-Stokes solver on an arbitrary masked domain. It uses a fractional-step (pressure-correction) method to advance an incompressible flow field in time.

### Core classes

| Class | Description |
|---|---|
| `ScalarProfile` | Base class that holds a boolean mask, computes interior/boundary cells, and tags regions with `MeshType` (WALL, INLET, OUTLET, FLOW) |
| `VelocityField` | Extends `ScalarProfile`; stores `vx` / `vy` components and advances them with an explicit finite-difference scheme (convection + diffusion + pressure gradient) |
| `SinkSource` | Builds the RHS divergence term (`b_field`) used by the pressure solver |
| `PressureField` | Solves the Poisson pressure equation iteratively (Gauss-Seidel-like) given the source term |

### Utilities

- `cfl()` — checks convective and diffusive CFL conditions and asserts stability.
- `get_safe_dt()` — computes a recommended time-step with a configurable safety margin.

## Status

> **This project is still in development.** Expect incomplete features, breaking changes, and rough edges.

## Requirements

- Python 3.10+
- `numpy`
- `matplotlib`
