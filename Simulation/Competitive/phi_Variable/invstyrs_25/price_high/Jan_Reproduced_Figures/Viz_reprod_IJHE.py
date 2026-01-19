# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 06:33:51 2026

@author: Mhdella
"""
# ============================================================
# Reproduce ORIGINAL OPGF Generation Mix Figures (Exact Style)
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Base directory (EDIT IF NEEDED)
# -----------------------------
BASE_DIR = r"C:\Users\Mhdella\Desktop\GB_IES_GTM\IJHE_GB_GTM\Output\Competitive\phi_Variable\invstyrs_25\price_high"

TIME_STEPS = "time_steps_13"
DEMAND_LEVEL = "Demand_level_Peak"
EXCEL_FILE = "Generation_results.xlsx"
SHEET_NAME = "OPGF Model"

# -----------------------------
# Investment scenarios
# -----------------------------
plan_scenarios = {
    '25GW': "Uniform +25GW/Tech.",
    'HiRES_HiH2': "High RES–High H2",
    'HiRES_LwH2': "High RES–Low H2",
    'LwBESS_HiH2': "High H2–Low BESS"
}

# -----------------------------
# Output folder
# -----------------------------
FIGURES_OUT = os.path.join(BASE_DIR, "Reproduced_Figures")
os.makedirs(FIGURES_OUT, exist_ok=True)

# -----------------------------
# Matplotlib style (match model)
# -----------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.dpi": 300
})

BAR_COLOR = "blue"

# -----------------------------
# Load OPGF results
# -----------------------------
def load_opgf_results(scenario_key):
    file_path = os.path.join(
        BASE_DIR,
        f"Case_{scenario_key}",
        TIME_STEPS,
        DEMAND_LEVEL,
        EXCEL_FILE
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

    df = pd.read_excel(file_path, sheet_name=SHEET_NAME)

    # Standardize column names
    df.columns = ["Type", "Generation_MW", "Percentage"]

    # Convert percentage column
    df["Percentage"] = (
        df["Percentage"]
        .astype(str)
        .str.replace("%", "")
        .astype(float)
    )

    # Remove negligible values
    df = df[df["Generation_MW"] > 1].copy()

    # Convert MW → GW
    df["Generation_GW"] = df["Generation_MW"] / 1000

    # Sort descending
    df.sort_values("Generation_GW", ascending=False, inplace=True)

    return df


# -----------------------------
# Plot single scenario (original style)
# -----------------------------
def plot_generation_mix(df, scenario_key):
    fig, ax = plt.subplots(figsize=(14, 7))

    bars = ax.bar(
        df["Type"],
        df["Generation_GW"],
        color=BAR_COLOR,
        edgecolor="black"
    )

    # Y-axis label (EXACT)
    ax.set_ylabel("Energy (GW)")

    # X-axis rotation (EXACT)
    ax.tick_params(axis="x", rotation=45)
    
    # Gridlines (as in original)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Percentage labels above bars
    for bar, pct in zip(bars, df["Percentage"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{pct:.2f}%",
            ha="center",
            va="bottom",
            fontsize=16
        )

    plt.tight_layout()

    out_path = os.path.join(
        FIGURES_OUT,
        f"Fig_main_OPGF_gens_{scenario_key}.png"
    )

    plt.savefig(out_path)
    plt.close(fig)

    print(f"Saved: {out_path}")


# -----------------------------
# Run for all scenarios
# -----------------------------
scenario_data = {}

for scenario_key in plan_scenarios.keys():
    print(f"Processing scenario: {scenario_key}")
    df_opgf = load_opgf_results(scenario_key)
    scenario_data[scenario_key] = df_opgf
    plot_generation_mix(df_opgf, scenario_key)


# -----------------------------
# Optional: combined 2×2 figure
# -----------------------------
fig, axs = plt.subplots(2, 2, figsize=(18, 12))
axs = axs.flatten()

for idx, (scenario_key, scenario_name) in enumerate(plan_scenarios.items()):
    df = scenario_data[scenario_key]
    ax = axs[idx]

    bars = ax.bar(
        df["Type"],
        df["Generation_GW"],
        color=BAR_COLOR,
        edgecolor="black"
    )

    ax.set_title(scenario_name)
    ax.set_ylabel("Energy (GW)")
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    for bar, pct in zip(bars, df["Percentage"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct:.2f}%",
            ha="center",
            va="bottom",
            fontsize=14
        )

plt.tight_layout()

combined_path = os.path.join(
    FIGURES_OUT,
    "Fig_all_simulations_OPGF_generation_mix.png"
)

plt.savefig(combined_path)
# plt.close(fig)

print(f"\nCombined figure saved to: {combined_path}")
