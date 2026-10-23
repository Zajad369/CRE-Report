"""
Kinetic Analysis of Benzene Decomposition (Task 2)
-------------------------------------------------
Determines reaction order (integral method), rate constant at 30 °C,
temperature extrapolation to 50 °C (Arrhenius), and reaction half-life.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

from report_style import print_section, print_table, set_plot_style, style_axes

set_plot_style()

# ==============================================================================
# 1. EXPERIMENTAL DATA & CONSTANTS
# ==============================================================================
time = np.arange(0, 105, 5)  # min (0 to 100 min)
c_benzene = np.array(
    [
        0.500,
        0.339,
        0.260,
        0.189,
        0.160,
        0.131,
        0.114,
        0.100,
        0.090,
        0.081,
        0.074,
        0.068,
        0.060,
        0.059,
        0.055,
        0.052,
        0.050,
        0.046,
        0.044,
        0.042,
        0.040,
    ]
)  # mol/L

M_BENZENE = 78.0  # g/mol
R_GAS = 8.314  # J / (mol * K)
EA_KCAL = 26.6  # kcal / mol
EA_JOULE = EA_KCAL * 4184.0  # J / mol (111294.4 J/mol)

# ==============================================================================
# 2. INTEGRAL METHOD LINEARIZATION (Order 0, 1, 2)
# ==============================================================================
models = {
    "Zero (0)": (c_benzene, "C(t)", "mol / (L*min)"),
    "First (1)": (np.log(c_benzene), "ln(C(t))", "1 / min"),
    "Second (2)": (1.0 / c_benzene, "1 / C(t)", "L / (mol*min)"),
}

results = []
regs = {}

for order, (y_data, form, unit) in models.items():
    reg = linregress(time, y_data)
    regs[order] = reg
    k_val = reg.slope if "Second" in order else -reg.slope
    results.append(
        {
            "Reaction Order": order,
            "Linear Form": form,
            "Slope": reg.slope,
            "Intercept": reg.intercept,
            "R^2": reg.rvalue**2,
            "Rate Constant k": k_val,
            "Unit of k": unit,
        }
    )

df_models = pd.DataFrame(results)
print_section("INTEGRAL METHOD LINEARIZATION RESULTS (T = 30 °C)", leading_newline=False)
print_table(df_models)

# Best fit parameters (Second-order reaction)
reg_2nd = regs["Second (2)"]
k1_30 = reg_2nd.slope  # L / (mol * min)
k1_30_s = k1_30 / 60.0  # L / (mol * s)

# ==============================================================================
# 3. TEMPERATURE EXTRAPOLATION & HALF-LIFE AT 50 °C
# ==============================================================================
T1 = 30.0 + 273.15  # 303.15 K
T2 = 50.0 + 273.15  # 323.15 K

# Integrated Arrhenius: ln(k2 / k1) = (Ea / R) * (1/T1 - 1/T2)
delta_inv_T = (1.0 / T1) - (1.0 / T2)
ln_ratio = (EA_JOULE / R_GAS) * delta_inv_T
ratio_k = np.exp(ln_ratio)
k2_50 = k1_30 * ratio_k  # L / (mol * min)
k2_50_s = k2_50 / 60.0  # L / (mol * s)

# Half-life at 50 °C for rho_0 = 100 g/L -> C_0 = rho_0 / M
rho_0 = 100.0  # g/L
C_0 = rho_0 / M_BENZENE  # mol/L (~1.282 mol/L)
t_half_min = 1.0 / (k2_50 * C_0)  # min
t_half_s = t_half_min * 60.0  # s

df_extrap = pd.DataFrame(
    [
        {
            "Parameter": "Rate constant at 30 °C (k1)",
            "Value": f"{k1_30:.4f}",
            "Unit": "L / (mol*min)",
        },
        {
            "Parameter": "Rate constant at 30 °C (k1)",
            "Value": f"{k1_30_s:.6f}",
            "Unit": "L / (mol*s)",
        },
        {
            "Parameter": "Ea / R",
            "Value": f"{EA_JOULE / R_GAS:.1f}",
            "Unit": "K",
        },
        {
            "Parameter": "Delta(1/T)",
            "Value": f"{delta_inv_T:.4e}",
            "Unit": "1 / K",
        },
        {"Parameter": "Ratio k2 / k1", "Value": f"{ratio_k:.2f}", "Unit": "-"},
        {
            "Parameter": "Rate constant at 50 °C (k2)",
            "Value": f"{k2_50:.3f}",
            "Unit": "L / (mol*min)",
        },
        {
            "Parameter": "Rate constant at 50 °C (k2)",
            "Value": f"{k2_50_s:.4f}",
            "Unit": "L / (mol*s)",
        },
        {
            "Parameter": "Initial concentration C_0",
            "Value": f"{C_0:.3f}",
            "Unit": "mol / L",
        },
        {
            "Parameter": "Half-life t_1/2 at 50 °C",
            "Value": f"{t_half_min:.2f}",
            "Unit": "min",
        },
        {
            "Parameter": "Half-life t_1/2 at 50 °C",
            "Value": f"{t_half_s:.1f}",
            "Unit": "s",
        },
    ]
)

print_section("EXTRAPOLATION & HALF-LIFE PARAMETERS")
print_table(df_extrap)

# ==============================================================================
# 4. PLOTS
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# 0-Order Plot
axes[0].plot(time, c_benzene, "o", color="#1f77b4", label="Data")
axes[0].plot(
    time,
    regs["Zero (0)"].intercept + regs["Zero (0)"].slope * time,
    "-",
    color="#1f77b4",
    label=f"Fit (R² = {regs['Zero (0)'].rvalue**2:.4f})",
)
axes[0].set_xlabel("Time $t$ / min")
axes[0].set_ylabel(r"$C_{\mathrm{benzene}}$ / $\mathrm{mol\,L^{-1}}$")
axes[0].set_title("Zero-Order Model")
axes[0].legend()

# 1-Order Plot
axes[1].plot(time, np.log(c_benzene), "s", color="#ff7f0e", label="Data")
axes[1].plot(
    time,
    regs["First (1)"].intercept + regs["First (1)"].slope * time,
    "-",
    color="#ff7f0e",
    label=f"Fit (R² = {regs['First (1)'].rvalue**2:.4f})",
)
axes[1].set_xlabel("Time $t$ / min")
axes[1].set_ylabel(r"$\ln(C_{\mathrm{benzene}}\,/\,\mathrm{mol\,L^{-1}})$")
axes[1].set_title("First-Order Model")
axes[1].legend()

# 2-Order Plot
axes[2].plot(time, 1.0 / c_benzene, "^", color="#2ca02c", label="Data")
axes[2].plot(
    time,
    reg_2nd.intercept + reg_2nd.slope * time,
    "-",
    color="#2ca02c",
    label=f"Fit: $k = {k1_30:.3f}$ (R² = {reg_2nd.rvalue**2:.4f})",
)
axes[2].set_xlabel("Time $t$ / min")
axes[2].set_ylabel(r"$1\,/\,C_{\mathrm{benzene}}$ / $\mathrm{L\,mol^{-1}}$")
axes[2].set_title("Second-Order Model (Best Fit)")
axes[2].legend()

style_axes(*axes)
plt.tight_layout()
plt.savefig("fig_task2_order.png", dpi=300)
plt.show()