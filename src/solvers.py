"""
Numerical solvers for the non-linear damped pendulum.
Equation: theta'' + (b/m) theta' + (g/L) sin(theta) = 0
(b is a linear-drag coefficient in kg/s, so the angular damping rate is b/m.)
"""

import numpy as np
from scipy.integrate import solve_ivp


class NonLinearPendulum:
    def __init__(self, m=1.0, L=1.0, b=0.2, g=9.81):
        self.m, self.L, self.b, self.g = m, L, b, g

    def derivatives(self, t, y):
        """State vector y = [theta, omega]."""
        theta, omega = y
        return [omega, -(self.b / self.m) * omega - (self.g / self.L) * np.sin(theta)]

    def total_energy(self, y):
        """Mechanical energy E = kinetic + potential (y has shape (2, N))."""
        theta, omega = np.atleast_1d(y[0]), np.atleast_1d(y[1])
        kinetic = 0.5 * self.m * (self.L * omega) ** 2
        potential = self.m * self.g * self.L * (1.0 - np.cos(theta))
        return kinetic + potential


def rk4_step(func, t, y, dt):
    """Classic 4th-order Runge-Kutta single step."""
    y = np.asarray(y, dtype=np.float64)
    k1 = np.asarray(func(t, y), dtype=np.float64)
    k2 = np.asarray(func(t + 0.5 * dt, y + 0.5 * dt * k1), dtype=np.float64)
    k3 = np.asarray(func(t + 0.5 * dt, y + 0.5 * dt * k2), dtype=np.float64)
    k4 = np.asarray(func(t + dt, y + dt * k3), dtype=np.float64)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_custom_rk4(sys, y0, t_span, dt):
    """Fixed-step RK4 driver. Time grid is built with linspace so the last
    point is exactly t_span[1] (no floating-point overshoot)."""
    n_steps = int(round((t_span[1] - t_span[0]) / dt))
    t_eval = np.linspace(t_span[0], t_span[1], n_steps + 1)
    y_sol = np.zeros((2, n_steps + 1))
    y_sol[:, 0] = y0
    for i in range(n_steps):
        y_sol[:, i + 1] = rk4_step(sys.derivatives, t_eval[i], y_sol[:, i], dt)
    return t_eval, y_sol


def simulate_scipy(sys, y0, t_span, t_eval, method="RK45", rtol=1e-10, atol=1e-12):
    """Wrapper around scipy solve_ivp with tight, adjustable tolerances."""
    sol = solve_ivp(sys.derivatives, t_span, y0, method=method,
                    t_eval=t_eval, rtol=rtol, atol=atol)
    return sol.t, sol.y