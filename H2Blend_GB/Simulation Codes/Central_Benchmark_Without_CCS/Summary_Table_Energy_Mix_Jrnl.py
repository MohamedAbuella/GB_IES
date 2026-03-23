# -*- coding: utf-8 -*-
"""
IJHE-ready Hydrogen Blending Energy Mix → Excel Export Version
Only generates summary Excel file (no figures).
"""

# ============================================================
# Imports
# ============================================================

import os
import pandas as pd
import numpy as np

# ============================================================
# USER SETTINGS
# ============================================================

# Res_folder = "CfD_GreyH2_Cooperative"
# Res_folder = "CfD_BlueH2_Cooperative"
Res_folder = "CfD_GreyH2_Central"


price_case = "price_high"
# price_case = "price_low"

# Root directory of GB_Blend_H2
ROOT_DIR = r"C:\Users\Mhdella\Desktop\GB_Blend_H2"

# Output folder
BASE_OUTPUT = os.path.join(
    ROOT_DIR,
    "Output",
    Res_folder
)

# ============================================================
# SAVE LOCATION
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

CASE_NAME = "Case_HiRES_HiH2"
TIME_STEPS = "time_steps_13"
DEMAND_LEVEL = "Demand_level_Peak"
EXCEL_FILE = "Generation_results.xlsx"
SHEET_NAME = "OPGF Model"

# ============================================================
# Load Results Function
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

    # Standardize columns
    df.columns = ["Type", "Generation_MW", "Percentage"]

    # Technology renaming for consistency
    if "Grey" in Res_folder:
        df["Type"] = df["Type"].replace({"Gas CCS": "Gas"})

    if "Blue" in Res_folder:
        df["Type"] = df["Type"].replace({
            "Gas CCS": "Gas+CCS",
            "Biomass": "BECCS",
            "G2G": "G2G+CCS"
        })

    # Clean percentage column
    df["Percentage"] = (
        df["Percentage"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .astype(float)
    )

    # Remove very small generation entries
    df = df[df["Generation_MW"] > 1].copy()

    # Convert MW → GW (≈ GWh assumption for annualized model output)
    df["Generation_GWh"] = df["Generation_MW"] / 1000

    df.sort_values("Generation_GWh", ascending=False, inplace=True)

    return df

# ============================================================
# Main Execution → Excel Summary Export
# ============================================================

def main():

    h2_folders = sorted([
        f for f in os.listdir(BASE_OUTPUT)
        if f.startswith("H2_prop_")
    ])

    scenario_data = {}
    all_tech_types = set()

    # --------------------------------------------------------
    # Collect scenario data
    # --------------------------------------------------------

    for h2_folder in h2_folders:

        df = load_opgf_results(h2_folder)

        if df is None:
            continue

        share_label = h2_folder.replace("H2_prop_", "")

        df_proc = df[[
            "Type",
            "Generation_GWh",
            "Percentage"
        ]].copy()

        df_proc.set_index("Type", inplace=True)

        scenario_data[share_label] = df_proc

        all_tech_types.update(df_proc.index.tolist())

    if len(scenario_data) == 0:
        print("No scenario data found.")
        return

    # --------------------------------------------------------
    # Build unified summary table
    # --------------------------------------------------------

    all_tech_types = sorted(list(all_tech_types))
    summary_df = pd.DataFrame(index=all_tech_types)

    sorted_shares = sorted(
        scenario_data.keys(),
        key=lambda x: float(x)
    )

    for share in sorted_shares:

        df_share = scenario_data[share]

        df_share = df_share.reindex(all_tech_types)

        summary_df[f"H2_{share}_GWh"] = df_share["Generation_GWh"]
        summary_df[f"H2_{share}_%"] = df_share["Percentage"]

    # --------------------------------------------------------
    # Save Excel Output
    # --------------------------------------------------------

    output_excel_path = os.path.join(
        SAVE_DIR,
        "OPGF_GenerationMix_Summary.xlsx"
    )

    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="OPGF_Summary")

    print("\n✅ Excel summary file generated successfully!")
    print(f"📂 Location:\n{output_excel_path}")


# ============================================================
# Run Script
# ============================================================

if __name__ == "__main__":
    main()