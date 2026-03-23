# -*- coding: utf-8 -*-
"""
IJHE-ready Summary Results Visualization
Hydrogen Blending – Simulation Summary & Policy Support Summary

Creates:
• Individual metric barplots
• Combined multi-metric plots
• Excel summary tables (Metrics as rows, H2 shares as columns)
• Separate folders per sheet for figures
• Summary tables saved in main folder to avoid confusion

Saved inside:
Output/CfD_GreyH2_Cooperative/Viz_Summary_Res_Jrnl/
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

# Res_folder = "CfD_GreyH2_Cooperative"
# Res_folder = "CfD_BlueH2_Cooperative"
Res_folder = "CfD_GreyH2_Central"



## Select price case

price_case = "price_high"
# price_case = "price_low"


ROOT_DIR = r"C:\Users\Mhdella\Desktop\GB_Blend_H2"

BASE_OUTPUT = os.path.join(
    ROOT_DIR,
    "Output",
    Res_folder
)

# ============================================================
# 📌 Dynamic Metric Selection (EDIT FREELY)
# ============================================================

SIM_METRICS_TO_PLOT = [
    "Total Demand [GWh]",
    "Total Energy Supply [GWh]",
    "Total CO2 Emissions [Tonnes]",
    "Total Operational Cost [m£]",
    "Net Present Value (NPV) [b£]",
    "Hydrogen Marginal Cost [£/MWh]",
    "P2G [GWh]",
    "G2G [GWh]"

]

POLICY_METRICS_TO_PLOT = [
    "Opex-based Support for Electricity [m£]",
    "Opex-based Support for Hydrogen [m£]",
    "CfD-based Support for Electricity [m£]",
    "CfD-based Support for Hydrogen [m£]"
]

# ============================================================
# Folder Structure
# ============================================================

MASTER_SAVE_DIR = os.path.join(
    BASE_OUTPUT,
    "Viz_Summary_Res_Jrnl",
    price_case
)

SIM_SAVE_DIR = os.path.join(MASTER_SAVE_DIR, "Simulation_Summary")
POLICY_SAVE_DIR = os.path.join(MASTER_SAVE_DIR, "Policy_Support_Summary")

# Folder for Excel tables (outside subfolders)
TABLES_SAVE_DIR = MASTER_SAVE_DIR
os.makedirs(SIM_SAVE_DIR, exist_ok=True)
os.makedirs(POLICY_SAVE_DIR, exist_ok=True)
os.makedirs(TABLES_SAVE_DIR, exist_ok=True)

# ============================================================
# IJHE Journal Style
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
# Load Summary Sheet
# ============================================================

def load_summary_sheet(h2_folder, sheet_name):

    file_path = os.path.join(
        BASE_OUTPUT,
        h2_folder,
        "PI_CfD_0_CO2_165",
        price_case,
        "All_simulations_results.xlsx"
    )

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = ["Metric", "Value"]
    return df

# ============================================================
# Collect Data Across All H2 Shares
# ============================================================

def collect_metrics(sheet_name, selected_metrics):

    data = {}

    h2_folders = sorted([
        f for f in os.listdir(BASE_OUTPUT)
        if f.startswith("H2_prop_")
    ])

    for h2_folder in h2_folders:
        df = load_summary_sheet(h2_folder, sheet_name)
        if df is None:
            continue

        h2_value = h2_folder.replace("H2_prop_", "")

        for metric in selected_metrics:
            value = df.loc[df["Metric"] == metric, "Value"]
            if not value.empty:
                data.setdefault(metric, {})[h2_value] = float(value.values[0])

    # DataFrame: rows = H2 share, columns = metrics
    df_out = pd.DataFrame(data)

    return df_out

# ============================================================
# Plot Individual Metric
# ============================================================

def plot_individual_metric(metric, df_metric, save_dir):

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    x = np.arange(len(df_metric.index))
    bars = ax.bar(
        x,
        df_metric.values,
        color="steelblue",
        edgecolor="black"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df_metric.index)
    ax.set_ylabel(metric)
    ax.set_xlabel("Hydrogen Blending Share")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()

    save_path = os.path.join(
        save_dir,
        f"{metric.replace('/', '_').replace(' ', '_')}"
    )

    fig.savefig(save_path + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(save_path + ".pdf", bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Combined Multi-Metric Plot
# ============================================================

def plot_combined(df_all, save_dir, title):

    fig, ax = plt.subplots(figsize=(10, 6))

    df_all.plot(
        kind="bar",
        ax=ax,
        edgecolor="black"
    )

    ax.set_xlabel("Hydrogen Blending Share")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.xticks(rotation=0)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "Combined_Metrics")

    fig.savefig(save_path + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(save_path + ".pdf", bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Save Summary Table (FIXED ORIENTATION, SHORT NAMES)
# ============================================================

def save_summary_excel(df_all, save_dir, sheet_name):

    # Transpose so:
    # Rows = Metrics
    # Columns = H2 shares
    df_table = df_all.transpose()

    # Short file names
    if sheet_name == "Simulation Summary":
        excel_file_name = "Simulation_Table.xlsx"
    else:
        excel_file_name = "Policy_Support_Table.xlsx"

    excel_path = os.path.join(save_dir, excel_file_name)

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df_table.to_excel(writer, sheet_name=sheet_name)

    print(f"Summary table saved: {excel_path}")

# ============================================================
# MAIN EXECUTION
# ============================================================

# -------- Simulation Summary --------
sim_df = collect_metrics("Simulation Summary", SIM_METRICS_TO_PLOT)

if not sim_df.empty:

    sim_df = sim_df.sort_index()

    for metric in sim_df.columns:
        plot_individual_metric(metric, sim_df[metric], SIM_SAVE_DIR)

    plot_combined(sim_df, SIM_SAVE_DIR, "Simulation Summary Metrics")
    save_summary_excel(sim_df, TABLES_SAVE_DIR, "Simulation Summary")


# -------- Policy Support Summary --------
policy_df = collect_metrics("Policy Support Summary", POLICY_METRICS_TO_PLOT)

if not policy_df.empty:

    policy_df = policy_df.sort_index()

    for metric in policy_df.columns:
        plot_individual_metric(metric, policy_df[metric], POLICY_SAVE_DIR)

    plot_combined(policy_df, POLICY_SAVE_DIR, "Policy Support Metrics")
    save_summary_excel(policy_df, TABLES_SAVE_DIR, "Policy Support Summary")


print("\nVisualization and summary tables generated successfully.")
