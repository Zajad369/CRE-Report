"""
Kinetic Analysis of Phenol Oxidation (Task 1)
---------------------------------------------
Evaluates reaction orders (phenol and CuSO4 catalyst)
and determines activation energy via Arrhenius linearization.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

from report_style import print_section, print_table, set_plot_style, style_axes

set_plot_style()

# ==============================================================================
# 0. EXPERIMENTAL DATASETS (Tables 1 & 2)
# ==============================================================================
time = np.arange(0, 21, 1)  # 0 to 20 min

# Table 1: Catalyst variation at T = 70 °C (C_phenol^0 = 50 mg/L, C_AP^0 = 5 g/L)
data_catalyst = {
    "t_min": time,
    25: np.array(
        [
            50.0,
            49.5,
            49.4,
            48.1,
            46.6,
            45.1,
            43.8,
            42.4,
            40.9,
            39.5,
            38.2,
            36.7,
            35.2,
            33.8,
            32.4,
            30.3,
            29.5,
            28.1,
            26.7,
            25.2,
            23.5,
        ]
    ),
    50: np.array(
        [
            50.0,
            49.0,
            48.8,
            47.0,
            45.2,
            42.7,
            41.6,
            39.8,
            38.0,
            36.3,
            34.3,
            32.7,
            30.9,
            29.1,
            27.3,
            24.6,
            23.7,
            21.9,
            20.1,
            18.3,
            16.5,
        ]
    ),
    100: np.array(
        [
            50.0,
            48.7,
            48.3,
            46.1,
            43.9,
            40.4,
            39.4,
            37.2,
            34.9,
            32.7,
            30.3,
            28.2,
            26.0,
            23.8,
            21.5,
            18.2,
            17.1,
            14.8,
            12.6,
            10.4,
            8.1,
        ]
    ),
    150: np.array(
        [
            50.0,
            49.6,
            47.0,
            44.5,
            41.9,
            37.2,
            36.8,
            34.2,
            31.6,
            29.1,
            25.7,
            23.9,
            21.4,
            18.8,
            16.2,
            11.6,
            11.1,
            8.5,
            5.9,
            3.4,
            1.8,
        ]
    ),
}

# Table 2: Temperature variation at C_CuSO4^0 = 100 mg/L
c_80_raw = [
    50.0,
    49.9,
    44.8,
    40.0,
    35.1,
    30.1,
    25.2,
    20.3,
    15.3,
    10.4,
    5.5,
    4.8,
    3.9,
    3.1,
    2.2,
    1.6,
]
c_80 = np.array(c_80_raw + [np.nan] * (len(time) - len(c_80_raw)))

data_temp = {
    "t_min": time,
    60: np.array(
        [
            50.0,
            49.9,
            49.4,
            48.5,
            47.7,
            46.8,
            46.0,
            45.2,
            44.3,
            43.5,
            42.7,
            41.9,
            41.0,
            40.2,
            39.4,
            38.6,
            37.7,
            36.9,
            36.1,
            35.3,
            34.3,
        ]
    ),
    65: np.array(
        [
            50.0,
            49.7,
            48.3,
            47.0,
            45.6,
            44.2,
            42.8,
            41.5,
            40.1,
            38.7,
            36.2,
            36.0,
            34.6,
            33.2,
            31.8,
            30.5,
            29.1,
            27.7,
            26.3,
            25.0,
            23.9,
        ]
    ),
    70: data_catalyst[100],
    80: c_80,
}

M_PHENOL = 94.0  # g/mol
R_GAS = 8.314  # J / (mol * K)

# ==============================================================================
# 1. DETERMINATION OF REACTION ORDER W.R.T. PHENOL (Integral Method)
# ==============================================================================
print_section(
    "1. DETERMINATION OF REACTION ORDER W.R.T. PHENOL (Integral Method)",
    leading_newline=False,
)

# Reference series: 70 °C, 100 mg/L CuSO4
c_ref_mg = data_catalyst[100]
c_ref_mol = c_ref_mg * 1e-3 / M_PHENOL  # convert mg/L to mol/L

# Models: Zero (C vs t), First (ln C vs t), Second (1/C vs t)
y_zero = c_ref_mol
y_first = np.log(c_ref_mol)
y_second = 1.0 / c_ref_mol

reg_zero = linregress(time, y_zero)
reg_first = linregress(time, y_first)
reg_second = linregress(time, y_second)

r2_df = pd.DataFrame(
    {
        "Reaction Order": ["Zero (0)", "First (1)", "Second (2)"],
        "Linear Form": ["C(t)", "ln(C(t))", "1 / C(t)"],
        "Slope": [reg_zero.slope, reg_first.slope, reg_second.slope],
        "Intercept": [
            reg_zero.intercept,
            reg_first.intercept,
            reg_second.intercept,
        ],
        "R^2": [reg_zero.rvalue**2, reg_first.rvalue**2, reg_second.rvalue**2],
    }
)
print_table(r2_df)

# Plotting the three linearization models
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Zero order plot
axes[0].plot(time, y_zero * 1e4, "o", color="#1f77b4", label="Data")
axes[0].plot(
    time,
    (reg_zero.intercept + reg_zero.slope * time) * 1e4,
    "-",
    color="#1f77b4",
    label=f"Fit (R² = {reg_zero.rvalue**2:.4f})",
)
axes[0].set_xlabel("Time $t$ / min")
axes[0].set_ylabel(r"$C_{\mathrm{phenol}}$ / $10^{-4}\,\mathrm{mol\,L^{-1}}$")
axes[0].set_title("Zero-Order Model")
axes[0].legend()

# First order plot
axes[1].plot(time, y_first, "s", color="#ff7f0e", label="Data")
axes[1].plot(
    time,
    reg_first.intercept + reg_first.slope * time,
    "-",
    color="#ff7f0e",
    label=f"Fit (R² = {reg_first.rvalue**2:.4f})",
)
axes[1].set_xlabel("Time $t$ / min")
axes[1].set_ylabel(r"$\ln(C_{\mathrm{phenol}}\,/\,\mathrm{mol\,L^{-1}})$")
axes[1].set_title("First-Order Model")
axes[1].legend()

# Second order plot
axes[2].plot(time, y_second, "^", color="#2ca02c", label="Data")
axes[2].plot(
    time,
    reg_second.intercept + reg_second.slope * time,
    "-",
    color="#2ca02c",
    label=f"Fit (R² = {reg_second.rvalue**2:.4f})",
)
axes[2].set_xlabel("Time $t$ / min")
axes[2].set_ylabel(r"$1\,/\,C_{\mathrm{phenol}}$ / $\mathrm{L\,mol^{-1}}$")
axes[2].set_title("Second-Order Model")
axes[2].legend()

style_axes(*axes)
plt.tight_layout()
plt.savefig("fig_order_phenol.png", dpi=300)
plt.show()

# ==============================================================================
# 2. DETERMINATION OF REACTION ORDER W.R.T. CuSO4
# ==============================================================================
print_section("2. DETERMINATION OF REACTION ORDER W.R.T. CuSO4")

cuso4_conc = [25, 50, 100, 150]  # mg/L
k_app_cuso4_list = []
r2_cuso4_runs = []

for c_cat in cuso4_conc:
    c_series_mol = data_catalyst[c_cat] * 1e-3 / M_PHENOL
    reg = linregress(time, c_series_mol)
    k_app_cuso4_list.append(-reg.slope)  # k_app in mol/(L*min)
    r2_cuso4_runs.append(reg.rvalue**2)

k_app_cuso4 = np.array(k_app_cuso4_list)
ln_k_app_cuso4 = np.log(k_app_cuso4)
ln_c_cuso4 = np.log(cuso4_conc)

# Linear regression: ln(k_app) = ln(K') + gamma * ln(C_CuSO4)
reg_cat = linregress(ln_c_cuso4, ln_k_app_cuso4)
gamma = reg_cat.slope

cuso4_table = pd.DataFrame(
    {
        "C0_CuSO4 [mg/L]": cuso4_conc,
        "ln(C0_CuSO4)": ln_c_cuso4,
        "k_app [mol/(L*min)]": k_app_cuso4,
        "ln(k_app)": ln_k_app_cuso4,
        "Fit R^2": r2_cuso4_runs,
    }
)
print_table(cuso4_table)
print(f"\nCuSO4 Reaction Order gamma = {gamma:.3f} (R^2 = {reg_cat.rvalue**2:.4f})")

# Linearization Plot for CuSO4
plt.figure(figsize=(6, 4.5))
plt.plot(ln_c_cuso4, ln_k_app_cuso4, "o", color="#d62728", label="Data points")
plt.plot(
    ln_c_cuso4,
    reg_cat.intercept + reg_cat.slope * ln_c_cuso4,
    "-",
    color="#d62728",
    label=f"Fit: slope $\gamma$ = {gamma:.2f} (R² = {reg_cat.rvalue**2:.4f})",
)
plt.xlabel(r"$\ln(C_{\mathrm{CuSO_4}}^{0}\,/\,\mathrm{mg\,L^{-1}})$")
plt.ylabel(r"$\ln(k_{\mathrm{app}}\,/\,\mathrm{mol\,L^{-1}\,min^{-1}})$")
plt.title(r"Determination of Reaction Order w.r.t. $\mathrm{CuSO_4}$")
style_axes(plt.gca())
plt.legend()
plt.tight_layout()
plt.savefig("fig_order_cuso4.png", dpi=300)
plt.show()

# ==============================================================================
# 3. ACTIVATION ENERGY (Arrhenius Analysis)
# ==============================================================================
print_section("3. ACTIVATION ENERGY (Arrhenius Analysis)")

temps_celsius = [60, 65, 70, 80]
temps_kelvin = np.array([T + 273.15 for T in temps_celsius])
inv_t = 1.0 / temps_kelvin

k_app_temp_list = []
r2_temp_list = []

# Combined Plot for All Temperature Regressions
plt.figure(figsize=(7, 5))
colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

for T, col in zip(temps_celsius, colors):
    c_raw = data_temp[T]

    # For 80 °C, restrict fit to linear regime (t <= 10 min) before phenol is depleted
    if T == 80:
        mask = (time <= 10) & (~np.isnan(c_raw))
    else:
        mask = ~np.isnan(c_raw)

    t_fit = time[mask]
    c_fit_mol = c_raw[mask] * 1e-3 / M_PHENOL

    reg_t = linregress(t_fit, c_fit_mol)
    k_app_temp_list.append(-reg_t.slope)
    r2_temp_list.append(reg_t.rvalue**2)

    # Plot measurements and fitted lines
    plt.plot(
        time[~np.isnan(c_raw)],
        c_raw[~np.isnan(c_raw)] * 1e-3 / M_PHENOL * 1e4,
        "o",
        color=col,
        label=f"$T = {T}\,^\circ\mathrm{{C}}$",
    )
    plt.plot(
        t_fit,
        (reg_t.intercept + reg_t.slope * t_fit) * 1e4,
        "-",
        color=col,
        alpha=0.8,
    )

plt.xlabel("Time $t$ / min")
plt.ylabel(r"$C_{\mathrm{phenol}}$ / $10^{-4}\,\mathrm{mol\,L^{-1}}$")
plt.title(
    "Zero-Order Kinetic Fits at Different Temperatures ($C_{\mathrm{CuSO_4}}^0 = 100\,\mathrm{mg\,L^{-1}}$)"
)
style_axes(plt.gca())
plt.legend()
plt.tight_layout()
plt.savefig("fig_temp_regressions.png", dpi=300)
plt.show()

k_app_temp = np.array(k_app_temp_list)
ln_k_app_temp = np.log(k_app_temp)

# Arrhenius Linear Regression: ln(k_app) = ln(A) - (E_a / R) * (1 / T)
reg_arr = linregress(inv_t, ln_k_app_temp)
slope_arr = reg_arr.slope
intercept_arr = reg_arr.intercept
r2_arr = reg_arr.rvalue**2

e_a_j = -slope_arr * R_GAS
e_a_kj = e_a_j / 1000.0
a_app = np.exp(intercept_arr)

# Table for Arrhenius data
arrhenius_table = pd.DataFrame(
    {
        "T [°C]": temps_celsius,
        "T [K]": temps_kelvin,
        "1/T [1/K]": inv_t,
        "k_app [mol/(L*min)]": k_app_temp,
        "ln(k_app)": ln_k_app_temp,
        "Regression R^2": r2_temp_list,
    }
)
print_table(arrhenius_table)

print(f"\nArrhenius Regression Slope (m = -E_a / R): {slope_arr:.2f} K")
print(f"Intercept ln(A_app):                       {intercept_arr:.4f}")
print(f"Activation Energy E_a:                     {e_a_kj:.2f} kJ/mol")
print(
    f"Pre-exponential factor A_app:              {a_app:.3e} mol/(L*min)"
)
print(f"Arrhenius R^2:                             {r2_arr:.4f}")

# Arrhenius Plot
plt.figure(figsize=(6, 4.5))
plt.plot(inv_t * 1e3, ln_k_app_temp, "o", color="#9467bd", label="Data points")
plt.plot(
    inv_t * 1e3,
    intercept_arr + slope_arr * inv_t,
    "-",
    color="#9467bd",
    label=f"Fit: $E_\\mathrm{{a}} = {e_a_kj:.2f}\\,\\mathrm{{kJ\\,mol^{{-1}}}}$\n(R² = {r2_arr:.4f})",
)
plt.xlabel(r"$1/T$ / $10^{-3}\,\mathrm{K^{-1}}$")
plt.ylabel(r"$\ln(k_{\mathrm{app}}\,/\,\mathrm{mol\,L^{-1}\,min^{-1}})$")
plt.title("Arrhenius Plot for Phenol Oxidation")
style_axes(plt.gca())
plt.legend()
plt.tight_layout()
plt.savefig("fig_arrhenius.png", dpi=300)
plt.show()