"""Pytest suite for physical invariants and numerical order of accuracy."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from solvers import NonLinearPendulum, simulate_custom_rk4, simulate_scipy


def test_undamped_energy_conservation():
    """With b = 0 the mechanical energy must stay constant."""
    system = NonLinearPendulum(m=1.0, L=1.0, b=0.0, g=9.81)
    y0 = [np.radians(30.0), 0.0]
    t_eval = np.linspace(0, 5, 500)
    _, y = simulate_scipy(system, y0, (0.0, 5.0), t_eval, method="RK45")
    E = system.total_energy(y)
    assert np.max(E) - np.min(E) < 1e-6


def test_energy_never_increases_with_damping():
    """With b > 0, dE/dt <= 0 (tiny tolerance for solver noise)."""
    system = NonLinearPendulum(m=1.0, L=1.0, b=0.5, g=9.81)
    y0 = [np.radians(60.0), 0.0]
    t_eval = np.linspace(0, 10, 1000)
    _, y = simulate_scipy(system, y0, (0.0, 10.0), t_eval, method="Radau")
    assert np.all(np.diff(system.total_energy(y)) <= 1e-9)


def test_rk4_is_fourth_order():
    """Halving dt should cut the error by about 2^4 = 16."""
    system = NonLinearPendulum()
    y0, t_span = [np.radians(45.0), 0.0], (0.0, 5.0)
    errors = []
    for dt in (0.1, 0.05, 0.025):
        t, y = simulate_custom_rk4(system, y0, t_span, dt)
        _, y_ref = simulate_scipy(system, y0, t_span, t,
                                  method="DOP853", rtol=1e-13, atol=1e-14)
        errors.append(np.sqrt(np.mean((y[0] - y_ref[0]) ** 2)))
    for e_coarse, e_fine in zip(errors, errors[1:]):
        assert 3.7 < np.log2(e_coarse / e_fine) < 4.5