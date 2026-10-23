"""
Van 't Hoff Differential Method for Reaction Kinetics (Problem 3)
Equimolar reaction: 1.00 M N-acetylcysteine + 1.00 M Iodoacetamide (1:1 stoichiometry)

Method:
1. Polynomial fit: C(t)
2. Analytical rate determination: r = -dC/dt
3. Van 't Hoff linearization: ln(r) = ln(k) + n * ln(C)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from report_style import print_section, print_table, set_plot_style, style_axes

set_plot_style()

# =============================================================================
# parameters to test different ranges/degrees
# =============================================================================
POLY_DEGREE = 5       # Degree of polynomial for fitting (e.g., 2, 3, or 4)

# Slice indices for the linear regression range (0 to 7):
# Example: 0 to 5 uses the first 5 data points (indices 0, 1, 2, 3, 4 -> t = 0 to 35 s)
FIT_START_IDX = 0     
FIT_END_IDX = 5       

# =============================================================================
# 1. EXPERIMENTAL DATA
# =============================================================================
time_exp = np.array([0.0, 5.0, 10.0, 25.0, 35.0, 50.0, 60.0])
conc_exp = np.array([1.00, 0.78, 0.62, 0.33, 0.21, 0.12, 0.09])

# =============================================================================
# 2. POLYNOMIAL COMPARISON (TABLE 1)
# =============================================================================
poly_comparison = []
ss_tot = np.sum((conc_exp - np.mean(conc_exp)) ** 2)

for deg in [1, 2, 3, 4, 5]:
    coeffs = np.polyfit(time_exp, conc_exp, deg)
    poly = np.poly1d(coeffs)
    r2 = 1.0 - (np.sum((conc_exp - poly(time_exp)) ** 2) / ss_tot)
    
    # Equation string formatting
    eq_parts = [f"{c:+.4e}*t^{deg-i}" if deg-i > 1 else (f"{c:+.4e}*t" if deg-i == 1 else f"{c:+.4f}") 
                for i, c in enumerate(coeffs)]
    poly_comparison.append({
        "Degree": deg,
        "R2": round(r2, 5),
        "Equation": "C(t) = " + " ".join(eq_parts)
    })

df_poly_comparison = pd.DataFrame(poly_comparison)

# =============================================================================
# 3. KINETIC CALCULATIONS (TABLE 2)
# =============================================================================
# Fit selected polynomial and determine analytical derivative r = -dC/dt
poly_chosen = np.poly1d(np.polyfit(time_exp, conc_exp, POLY_DEGREE))
poly_deriv = np.polyder(poly_chosen)

c_poly = poly_chosen(time_exp)
rate_poly = -poly_deriv(time_exp)
ln_c = np.log(c_poly)
ln_r = np.log(rate_poly)

# Mask for selected range
mask = np.zeros(len(time_exp), dtype=bool)
mask[FIT_START_IDX:FIT_END_IDX] = True

df_calc = pd.DataFrame({
    "Time [s]": time_exp,
    "c_real [mol/L]": conc_exp,
    "c_polynom [mol/L]": c_poly,
    "r [mol/(L*s)]": rate_poly,
    "ln(c)": ln_c,
    "ln(r)": ln_r,
})

# =============================================================================
# 4. LINEAR REGRESSION & FINAL PARAMETERS (TABLE 3)
# =============================================================================
slope, intercept, r_val, _, _ = stats.linregress(ln_c[mask], ln_r[mask])

n_order = slope
k_const = np.exp(intercept)
r2_lin = r_val ** 2

df_params = pd.DataFrame([
    {"Parameter": "Reaction Order (n)", "Value": f"{n_order:.4f}", "Unit": "-"},
    {"Parameter": "Rate Constant (k)", "Value": f"{k_const:.4e}", "Unit": f"(L/mol)^{n_order - 1:.2f} * s^-1"},
    {"Parameter": "Fit Quality (R2)", "Value": f"{r2_lin:.5f}", "Unit": "-"}
])

# =============================================================================
# 5. CONSOLE OUTPUT
# =============================================================================
print_section("TABLE 1: POLYNOMIAL FIT COMPARISON")
print_table(df_poly_comparison)

print_section("TABLE 2: CALCULATION DATASET")
print_table(df_calc, formatters={
    "Time [s]": "{:.1f}".format,
    "c_real [mol/L]": "{:.4f}".format,
    "c_polynom [mol/L]": "{:.4f}".format,
    "r [mol/(L*s)]": "{:.5e}".format,
    "ln(c)": "{:.4f}".format,
    "ln(r)": "{:.4f}".format,
})

print_section(
    f"TABLE 3: FINAL PARAMETERS (Points {FIT_START_IDX} to {FIT_END_IDX - 1} included)"
)
print_table(df_params)

# =============================================================================
# 6. PLOTS
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Concentration curve with single selected polynomial
t_line = np.linspace(time_exp.min(), time_exp.max(), 200)
axes[0].scatter(
    time_exp, conc_exp, color="#1f77b4", s=40, zorder=5, label="Raw Data"
)
axes[0].plot(
    t_line,
    poly_chosen(t_line),
    color="#1f77b4",
    linestyle="--",
    label=f"Poly Fit (Degree {POLY_DEGREE})",
)
axes[0].set_title("Concentration vs. Time")
axes[0].set_xlabel("Time $t$ [s]")
axes[0].set_ylabel("Concentration $C$ [mol/L]")
axes[0].legend()

# Plot 2: Van 't Hoff linear plot
axes[1].scatter(
    ln_c[mask],
    ln_r[mask],
    color="#1f77b4",
    s=50,
    zorder=5,
    label="Included Points",
)
if np.any(~mask):
    axes[1].scatter(
        ln_c[~mask],
        ln_r[~mask],
        color="#ff7f0e",
        facecolors="none",
        s=60,
        linewidth=1.5,
        zorder=5,
        label="Excluded Points",
    )

x_fit = np.linspace(ln_c[mask].min() - 0.1, ln_c[mask].max() + 0.1, 100)
y_fit = n_order * x_fit + intercept
axes[1].plot(
    x_fit,
    y_fit,
    color="#1f77b4",
    linestyle="-",
    label=f"Fit: $\\ln(r) = {n_order:.2f}\\,\\ln(C) {intercept:+.2f}$",
)

axes[1].set_title("Van 't Hoff Plot: $\\ln(r)$ vs. $\\ln(C)$")
axes[1].set_xlabel("$\\ln(C)$")
axes[1].set_ylabel("$\\ln(r)$")
axes[1].legend()

style_axes(*axes)
plt.tight_layout()
plt.show()