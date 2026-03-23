# -*- coding: utf-8 -*-
"""
IJHE-ready Hydrogen Blending Energy Mix Visualization
Saved inside main results folder:
Output/CfD_GreyH2_Cooperative/Viz_Barplots_Jrnl/
"""

# ============================================================
# Imports
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# USER SETTINGS
# ============================================================

Res_folder = "CfD_GreyH2_Cooperative"
# Res_folder = "CfD_BlueH2_Cooperative"


## Select price case

price_case = "price_high"
# price_case = "price_low"


### CASE_NAME = "Case_25GW"
CASE_NAME = "Case_HiRES_HiH2"



# Root directory of GB_Blend_H2
ROOT_DIR = r"C:\Users\Mhdella\Desktop\GB_Blend_H2"

# Output folder containing hydrogen blending results
BASE_OUTPUT = os.path.join(
    ROOT_DIR,
    "Output",
     Res_folder
)

# ============================================================
# ✅ NEW SAVE LOCATION (INSIDE MAIN RESULTS FOLDER)
# ============================================================

SAVE_DIR = os.path.join(
    BASE_OUTPUT,
    "Viz_Barplots_Jrnl",
    price_case
)

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# Constants
# ============================================================

# CASE_NAME = "Case_HiRES_HiH2"
TIME_STEPS = "time_steps_13"
DEMAND_LEVEL = "Demand_level_Peak"
EXCEL_FILE = "Generation_results.xlsx"
SHEET_NAME = "OPGF Model"

# ============================================================
# IJHE Style
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 14,
    "figure.dpi": 300,
    "axes.linewidth": 1.2
})

# ============================================================
# Technology Colors
# ============================================================

if "Grey" in Res_folder:
    tech_colors = {
        "Wind": "green",
        "PV": "orange",
        "Nuclear": "magenta",
        "P2G": "lime",
        "G2P": "olive",
        "G2G": "blue",
        "Gas": "brown",
        "Biomass": "red",
        "Geothermal": "grey",
        "Hydropower": "cyan",
        "BESS": "black",
        "H2-Storage": "purple"
    }


if "Blue" in Res_folder:
    tech_colors = {
        "Wind": "green",
        "PV": "orange",
        "Nuclear": "magenta",
        "P2G": "lime",
        "G2P": "olive",
        "G2G+CCS": "blue",
        "Gas+CCS": "brown",
        "BECCS": "red",
        "Geothermal": "grey",
        "Hydropower": "cyan",
        "BESS": "black",
        "H2-Storage": "purple"
    }

# ============================================================
# Load OPGF Results
# ============================================================

def load_opgf_results(h2_folder):

    file_path = os.path.join(
        BASE_OUTPUT,
        h2_folder,
        "PI_CfD_0_CO2_165",
        price_case,
        CASE_NAME,
        TIME_STEPS,
        DEMAND_LEVEL,
        EXCEL_FILE
    )

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    df = pd.read_excel(file_path, sheet_name=SHEET_NAME)
    df.columns = ["Type", "Generation_MW", "Percentage"]
    
    if "Grey" in Res_folder:
        df["Type"] = df["Type"].replace({"Gas CCS": "Gas"})
        
    if "Blue" in Res_folder:
        df["Type"] = df["Type"].replace({"Gas CCS": "Gas+CCS"})
        df["Type"] = df["Type"].replace({"Biomass": "BECCS"})
        df["Type"] = df["Type"].replace({"G2G": "G2G+CCS"})


    df["Percentage"] = (
        df["Percentage"].astype(str).str.replace("%", "").astype(float)
    )

    df = df[df["Generation_MW"] > 1].copy()
    df["Generation_GW"] = df["Generation_MW"] / 1000
    df.sort_values("Generation_GW", ascending=False, inplace=True)
    
    
    return df


# ============================================================
# Single Scenario Plot
# ============================================================

def plot_generation_mix(df, h2_folder):

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    x_pos = np.arange(len(df))
    bar_colors = [tech_colors.get(t, "grey") for t in df["Type"]]

    bars = ax.bar(
        x_pos,
        df["Generation_GW"],
        width=0.65,
        color=bar_colors,
        edgecolor="black",
        linewidth=1.0
    )

    ax.set_xticks([])
    ax.set_xlabel("Allocated Energy Mix")
    ax.set_ylabel("Energy (GWh)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for bar, pct in zip(bars, df["Percentage"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()
    
    save_path = os.path.join(SAVE_DIR,f"Fig_OPGF_GenerationMix_{h2_folder}_{price_case}")


    if "Blue" in Res_folder:
        save_path = os.path.join(SAVE_DIR,f"Fig_OPGF_GenerationMix_{h2_folder}_{price_case}_CCS")


    fig.savefig(save_path + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(save_path + ".pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")


# ============================================================
# Combined Comparison Plot
# ============================================================

def plot_combined(all_data):

    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    axs = axs.flatten()

    for idx, (h2_folder, df) in enumerate(all_data.items()):
        ax = axs[idx]

        x_pos = np.arange(len(df))
        bar_colors = [tech_colors.get(t, "grey") for t in df["Type"]]

        bars = ax.bar(
            x_pos,
            df["Generation_GW"],
            color=bar_colors,
            edgecolor="black"
        )

        ax.set_title(f"H2 Share: {h2_folder.replace('H2_prop_', '')}")
        ax.set_ylabel("Energy (GWh)")
        ax.set_xticks([])
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        for bar, pct in zip(bars, df["Percentage"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10
            )

    plt.tight_layout()

    combined_path = os.path.join(SAVE_DIR,f"Fig_All_H2_Scenarios_{price_case}")
    
    if "Blue" in Res_folder:
        combined_path = os.path.join(SAVE_DIR, f"Fig_All_H2_Scenarios_{price_case}_CCS")


    fig.savefig(combined_path + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(combined_path + ".pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nCombined figure saved: {combined_path}")


# ============================================================
# Legend
# ============================================================

def save_legend():

    fig, ax = plt.subplots(figsize=(8, 1.2))
    ax.axis("off")

    handles = []
    labels = []

    for tech, color in tech_colors.items():
        handles.append(
            plt.Line2D([], [], color=color, marker='s',
                       markersize=10, linestyle='None')
        )
        labels.append(tech)

    ax.legend(handles, labels, ncol=6, loc="center",
              frameon=False, fontsize=12)

    legend_path = os.path.join(SAVE_DIR, "Legend_Technologies")
    
    if "Blue" in Res_folder:
        legend_path = os.path.join(SAVE_DIR, "Legend_Technologies_CCS")

    fig.savefig(legend_path + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(legend_path + ".pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Legend saved: {legend_path}")


# ============================================================
# Main Execution
# ============================================================

all_results = {}

h2_folders = sorted([
    f for f in os.listdir(BASE_OUTPUT)
    if f.startswith("H2_prop_")
])

for h2_folder in h2_folders:

    df = load_opgf_results(h2_folder)

    if df is not None:
        all_results[h2_folder] = df
        plot_generation_mix(df, h2_folder)

if len(all_results) > 0:
    plot_combined(all_results)
    save_legend()

print("\nVisualization completed successfully.")
