"""
Time-step refinement sweep and empirical order of convergence for RK4.
Reference: DOP853 (8th order) at rtol=1e-13, evaluated on the SAME time grid
as each RK4 run, so no resampling or index rounding is needed.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from solvers import NonLinearPendulum, simulate_custom_rk4, simulate_scipy


def run_convergence_study():
    sys = NonLinearPendulum(m=1.0, L=1.0, b=0.2, g=9.81)
    y0 = [np.radians(45.0), 0.0]
    t_span = (0.0, 10.0)
    dt_list = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    l2_errors, linf_errors = [], []

    print("--- Convergence Analysis ---")
    for dt in dt_list:
        t_sim, y_sim = simulate_custom_rk4(sys, y0, t_span, dt)
        _, y_ref = simulate_scipy(sys, y0, t_span, t_sim,
                                  method="DOP853", rtol=1e-13, atol=1e-14)
        err = np.abs(y_sim[0] - y_ref[0])
        l2_errors.append(np.sqrt(np.mean(err ** 2)))
        linf_errors.append(np.max(err))
        print(f"dt = {dt:.5f} s | L2 = {l2_errors[-1]:.4e} | Linf = {linf_errors[-1]:.4e}")

    for i in range(len(dt_list) - 1):
        p = np.log2(l2_errors[i] / l2_errors[i + 1])
        print(f"Refinement {i + 1}: empirical order p = {p:.2f}")

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.loglog(dt_list, l2_errors, "o-", label=r"$L_2$ error")
    plt.loglog(dt_list, linf_errors, "s--", label=r"$L_\infty$ error")
    ref = [l2_errors[-1] * (dt / dt_list[-1]) ** 4 for dt in dt_list]
    plt.loglog(dt_list, ref, "k:", label=r"slope 4 reference")
    plt.xlabel(r"Time step $\Delta t$ (s)")
    plt.ylabel("Error norm")
    plt.title("RK4 temporal convergence (non-linear damped pendulum)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/convergence_plot.png", dpi=300)
    plt.close()

    t_eval = np.linspace(0, 10, 1000)
    _, y_rk45 = simulate_scipy(sys, y0, t_span, t_eval, method="RK45")
    _, y_radau = simulate_scipy(sys, y0, t_span, t_eval, method="Radau")
    E_rk45, E_radau = sys.total_energy(y_rk45), sys.total_energy(y_radau)
    print(f"Max |E_RK45 - E_Radau| = {np.max(np.abs(E_rk45 - E_radau)):.3e} J")

    plt.figure(figsize=(8, 5))
    plt.plot(t_eval, E_rk45, label="RK45 energy")
    plt.plot(t_eval, E_radau, "r--", label="Radau energy")
    plt.xlabel("Time (s)")
    plt.ylabel("Total mechanical energy (J)")
    plt.title(r"Energy decay check ($dE/dt \leq 0$)")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/energy_dissipation.png", dpi=300)
    plt.close()
    print("Plots saved to results/")


if __name__ == "__main__":
    run_convergence_study()