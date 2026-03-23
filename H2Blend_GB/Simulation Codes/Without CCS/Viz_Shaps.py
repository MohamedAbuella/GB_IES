# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 19:26:51 2026

@author: Mhdella
"""

import pandas as pd
import numpy as np
import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib as mpl


# ===============================
# USER INPUTS
# ===============================

Res_folder = "CfD_GreyH2_Cooperative"
# Res_folder = "CfD_BlueH2_Cooperative"
# Res_folder = "CfD_GreyH2_Central"


# price_scenario = "price_high"
price_scenario = "price_low"  


h2_blends = {
    # "0%": "H2_prop_0",
    # "10%": "H2_prop_0.1",
    "20%": "H2_prop_0.2",
    "100%": "H2_prop_1.0"
}

sheet_name = "Shapley Values"

# ===============================
# BASE PATH: relative to script location
# ===============================
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

base_path = os.path.join(script_dir, "Output", Res_folder)

# ===============================
# Initialize DataFrame
# ===============================
shapley_table = pd.DataFrame()

for blend_label, blend_folder in h2_blends.items():
    excel_path = os.path.join(
        base_path,
        blend_folder,
        "PI_CfD_0_CO2_165",
        price_scenario,
        "Case_HiRES_HiH2",
        "time_steps_13",
        "Demand_level_Peak",
        "Simulation_results.xlsx"
    )

    if not os.path.isfile(excel_path):
        print(f"ERROR: File not found: {excel_path}")
        continue

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df = df[['Type', 'Shapley Value']]
    df = df.rename(columns={'Shapley Value': blend_label})

    if shapley_table.empty:
        shapley_table = df
    else:
        shapley_table = pd.merge(shapley_table, df, on='Type')

# Sort by 100% H2
shapley_table = shapley_table.sort_values(by='100%', ascending=False).reset_index(drop=True)


# ===============================
# Normalize positive and negative values to sum = 1 per column
# ===============================
norm_positive = shapley_table.copy()
norm_negative = shapley_table.copy()

for col in h2_blends.keys():
    # Positive Shap normalization
    pos_sum = norm_positive[col][norm_positive[col] > 0].sum()
    if pos_sum != 0:
        norm_positive[col] = norm_positive[col].apply(lambda x: x / pos_sum if x > 0 else 0)
    
    # Negative Shap normalization (absolute sum = 1)
    neg_sum = abs(norm_negative[col][norm_negative[col] < 0].sum())
    if neg_sum != 0:
        norm_negative[col] = norm_negative[col].apply(lambda x: abs(x) / neg_sum if x < 0 else 0)

# ===============================
# Save to Excel with three sheets
# ===============================
output_folder = os.path.join(base_path, "Shaps", price_scenario)
os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(output_folder, "Shapley_Values_Comparison.xlsx")

with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    shapley_table.to_excel(writer, sheet_name="Shap_Values", index=False)
    norm_positive.to_excel(writer, sheet_name="Norm_Positive_Shaps", index=False)
    norm_negative.to_excel(writer, sheet_name="Norm_Negative_Shaps", index=False)

print(f"\nShapley values with normalized sheets saved to:\n{output_file}")




# ---------------------------
# Horizontal bar plot for H2 Prop = 100%
# optimized for 50% width subfigure
# ---------------------------

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],

    # Increased fonts
    "font.size": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
})

# Read normalized sheets
pos_shap = pd.read_excel(output_file, sheet_name="Norm_Positive_Shaps")
neg_shap = pd.read_excel(output_file, sheet_name="Norm_Negative_Shaps")

# Merge sheets
h2_100 = pos_shap[['Type','100%']].merge(
    neg_shap[['Type','100%']],
    on='Type',
    suffixes=('_pos','_neg')
)

# Compute signed normalized contribution
h2_100['Contribution'] = (h2_100['100%_pos'] - h2_100['100%_neg']) * 100

# Rename technologies
label_map = {
    'p2g': 'P2G',
    'g2p': 'G2P',
    'g2g': 'G2G',
    'PV': 'PV Solar',
    'g2p(Fuel Cell)': 'G2P (Fuel Cell)',
    'g2p(H2-CCGT)': 'G2P (H2-CCGT)',
    'g2p(H2-OCGT)': 'G2P (H2-OCGT)',
    'Hydro reservoir': 'Hydro Reservoir',
    'Gas CCS': 'Gas CCS'
}

h2_100['Type'] = h2_100['Type'].replace(label_map)

# Sort bars
h2_100 = h2_100.sort_values(by='Contribution', ascending=True)

# Colors
colors = ['#1e68a5' if v >= 0 else '#e74c3c' for v in h2_100['Contribution']]

# ---------------------------
# Plot
# ---------------------------

n = len(h2_100)

# Larger height for thicker bars and spacing
fig_height = n * 0.6
fig, ax = plt.subplots(figsize=(7, fig_height), facecolor='white')

bars = ax.barh(
    range(n),
    h2_100['Contribution'],
    color=colors,
    height=0.9
)

# Remove y-axis ticks
ax.set_yticks([])

# X-axis limits
max_val = max((h2_100['Contribution']))
min_val = min((h2_100['Contribution']))
ax.set_xlim(max_val*1.0, -min_val*1.0)

x_max = np.ceil(max_val / 10) * 10
x_min = np.floor(min_val / 10) * 10

ax.set_xlim(x_min, x_max)

ticks = np.arange(x_min, x_max + 10, 10)


ax.set_xticks(ticks)
ax.set_xticklabels([f"{int(t)}%" for t in ticks])

# Spine settings
ax.spines['bottom'].set_visible(True)
ax.spines['bottom'].set_linewidth(1.2)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.set_xlabel('Shapley Value Contribution (%)')

# Zero reference line
ax.axvline(0, color='black', linewidth=1.2)

# Grid
ax.grid(axis='x', linestyle='--', alpha=0.35)

label_offset = 1.2
name_offset = 1.0

for bar, val, tech in zip(bars, h2_100['Contribution'], h2_100['Type']):

    abs_val = abs(val)

    if abs_val < 1:
        pct = f"{abs_val:.2f}"
    else:
        pct = f"{round(abs_val)}"

    y = bar.get_y() + bar.get_height()/2

    if val >= 0:

        ax.text(
            -name_offset,
            y,
            tech,
            ha='right',
            va='center',
            fontsize=18
        )

        ax.text(
            val + label_offset,
            y,
            f"{pct}%",
            ha='left',
            va='center',
            fontsize=18
        )

    else:

        ax.text(
            name_offset,
            y,
            tech,
            ha='left',
            va='center',
            fontsize=18
        )

        ax.text(
            val - label_offset,
            y,
            f"{pct}%",
            ha='right',
            va='center',
            fontsize=18
        )

# Layout tuned for subfigure scaling
plt.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.18)

# Save plot
barplot_pdf = os.path.join(output_folder, "Shapley_Values_Bar_H2_100.pdf")

fig.savefig(
    barplot_pdf,
    dpi=600,
    bbox_inches='tight',
    facecolor='white'
)

print(f"\nBarplot saved as PDF:\n{barplot_pdf}")

plt.show()