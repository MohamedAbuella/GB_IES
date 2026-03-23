# -*- coding: utf-8 -*-
"""
Policy support allocation based on negative Shapley values
Energy and support distributed inversely proportional to normalized negative Shapley values
Electricity and Hydrogen prices applied based on scenario
"""

import pandas as pd
import os

# ===============================
# USER INPUTS
# ===============================

SA_folder = "PI_CfD_0_CO2_165"
Res_folder = "CfD_GreyH2_Cooperative"
price_scenario = "price_low"

# ===============================
# ELECTRICITY AND HYDROGEN PRICES
# ===============================

if price_scenario == "price_low":
    Ele_P = 50
    H2_P = 75
else:
    Ele_P = 100
    H2_P = 150

# ===============================
# BASE PATH
# ===============================

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

base_path = os.path.join(script_dir, "Output", Res_folder)

# ===============================
# FILE PATHS
# ===============================

gen_mix_file = os.path.join(
    base_path, "Viz_Barplots_Jrnl", price_scenario, "OPGF_GenerationMix_Summary.xlsx"
)

policy_file = os.path.join(
    base_path, "Viz_Summary_Res_Jrnl", price_scenario, "Policy_Support_Table.xlsx"
)

shap_file = os.path.join(
    base_path, "Shaps", price_scenario, "Shapley_Values_Comparison.xlsx"
)

# ===============================
# OUTPUT FILE
# ===============================

output_folder = os.path.dirname(policy_file)
os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(output_folder, "Policy_Support_By_Technology.xlsx")

# ===============================
# LOAD DATA
# ===============================

gen_mix = pd.read_excel(gen_mix_file, index_col=0)
policy = pd.read_excel(policy_file, index_col=0)

shap_values = pd.read_excel(shap_file, sheet_name="Shap_Values")
norm_neg_shap = pd.read_excel(shap_file, sheet_name="Norm_Negative_Shaps")

policy.columns = policy.columns.astype(str)

# ===============================
# TECHNOLOGY NAME MAPPING
# ===============================

tech_map = {
    "P2G": ["p2g"],
    "G2G": ["g2g"],
    "G2P": ["g2p(H2-CCGT)", "g2p(H2-OCGT)", "g2p(Fuel Cell)"],
    "PV": ["PV"],
    "Wind": ["Onshore Wind", "Offshore Wind"],
    "BESS": ["Storage"],
    "Hydropower": ["Hydro reservoir", "Hydro ROR"]
}

# ===============================
# SHAPLEY TO OPGF ROW MAPPING
# ===============================

energy_row_map = {
    "Storage": ["BESS", "H2-Storage"],
    "Hydro reservoir": ["Hydropower"],
    "Hydro ROR": ["Hydropower"],
    "p2g": ["P2G"],
    "g2p(H2-CCGT)": ["G2P"],
    "g2p(H2-OCGT)": ["G2P"],
    "g2p(Fuel Cell)": ["G2P"],
    "PV": ["PV"],
    "Onshore Wind": ["Wind"],
    "Offshore Wind": ["Wind"],
    "g2g": ["G2G"]
}

# ===============================
# NEGATIVE SHAPLEY TECHNOLOGIES
# ===============================

neg_shap = shap_values[shap_values["100%"] < 0]
neg_types = neg_shap["Type"].tolist()

norm_neg = norm_neg_shap.set_index("Type")["100%"]

# ===============================
# BLENDING LEVELS
# ===============================

blend_levels = ["0", "0.1", "0.2", "1.0"]

# ===============================
# CREATE EXCEL WRITER
# ===============================

writer = pd.ExcelWriter(output_file, engine="xlsxwriter")

# ===============================
# MAIN LOOP
# ===============================

for b in blend_levels:

    mix_col_pct = "H2_" + b + "_%"
    mix_col_gwh = "H2_" + b + "_GWh"

    if mix_col_pct not in gen_mix.columns:
        continue

    mix_percent = gen_mix[mix_col_pct]
    active_mix = mix_percent[mix_percent > 0.1]

    eligible = {}
    energy_alloc = {}
    tech_groups = {}

# ===============================
# IDENTIFY ELIGIBLE TECHS
# ===============================

    for mix_tech in active_mix.index:

        if mix_tech in tech_map:

            shap_techs = tech_map[mix_tech]

            neg_shap_techs = [st for st in shap_techs if st in neg_types]

            if not neg_shap_techs:
                continue

            tech_groups[mix_tech] = neg_shap_techs

            for st in neg_shap_techs:
                eligible[st] = norm_neg[st]

# ===============================
# ENERGY ALLOCATION
# ===============================

    for mix_tech, subtechs in tech_groups.items():

        rows = energy_row_map[subtechs[0]]

        total_gwh = sum([gen_mix.loc[r, mix_col_gwh] for r in rows])

        if len(subtechs) > 1:

            inv_shap = 1 / pd.Series(norm_neg[subtechs])
            shares = inv_shap / inv_shap.sum()

            for st in subtechs:
                energy_alloc[st] = total_gwh * shares[st]

        else:

            energy_alloc[subtechs[0]] = total_gwh

# ===============================
# SUPPORT ALLOCATION
# ===============================

    eligible_series = pd.Series(eligible)

    total_support = policy.loc[
        "Opex-based Support for Electricity [m£]", b
    ]

    support_alloc = {}

    for mix_tech, subtechs in tech_groups.items():

        group_shap_sum = eligible_series[subtechs].sum()

        group_support = total_support * (group_shap_sum / eligible_series.sum())

        if len(subtechs) > 1:

            inv_shap = 1 / pd.Series(norm_neg[subtechs])
            shares = inv_shap / inv_shap.sum()

            for st in subtechs:
                support_alloc[st] = group_support * shares[st]

        else:

            support_alloc[subtechs[0]] = group_support

# ===============================
# CALCULATE PROFITS
# ===============================

    total_profit = []

    for tech in eligible_series.index:

        if tech in ["p2g","g2g","g2p(H2-CCGT)","g2p(H2-OCGT)","g2p(Fuel Cell)"]:
            total_profit.append(energy_alloc[tech] * H2_P / 1e3)
        else:
            total_profit.append(energy_alloc[tech] * Ele_P / 1e3)

# ===============================
# BUILD RESULT TABLE
# ===============================

    result = pd.DataFrame({

        "Technology": eligible_series.index,
        "Normalized_Neg_Shap": eligible_series.values,
        "Energy_Allocation_GWh": [energy_alloc[t] for t in eligible_series.index],
        "Support_Deficit_(Profit-Cost)_m£": [support_alloc[t] for t in eligible_series.index],
        "Total_Profit_m£": total_profit

    })

    result = result.sort_values(
        "Support_Deficit_(Profit-Cost)_m£",
        ascending=False
    )

# ===============================
# TOTAL ROW
# ===============================

    total_row = pd.DataFrame({

        "Technology": ["Total"],
        "Normalized_Neg_Shap": [result["Normalized_Neg_Shap"].sum()],
        "Energy_Allocation_GWh": [result["Energy_Allocation_GWh"].sum()],
        "Support_Deficit_(Profit-Cost)_m£": [result["Support_Deficit_(Profit-Cost)_m£"].sum()],
        "Total_Profit_m£": [result["Total_Profit_m£"].sum()]

    })

    result = pd.concat([result, total_row], ignore_index=True)

# ===============================
# SAVE SHEET
# ===============================

    sheet_name = "Blend_" + b

    result.to_excel(writer, sheet_name=sheet_name, index=False)

# ===============================
# SAVE FILE
# ===============================

writer.close()

print("\nPolicy support file saved successfully:")
print(output_file)