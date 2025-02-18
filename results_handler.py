# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 19:40:01 2025

@author: Mhdella
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def format_results(results, objective):
    
    formatted_results = {}
    
    [results.pop(key, None) for key in ["Maximum generation capacity", "Losses"]]
    
    
    for key, value in results.items():
        if abs(value) >= 1e6:  # Convert to GW if >= 1,000,000 MW
            formatted_results[key] = f"{value / 1e3:.2f} GW"
        elif abs(value) >= 1e3:  # Convert to GWh if >= 1,000 MW
            formatted_results[key] = f"{value / 1e3:.2f} GWh"
        else:  # Keep in MWh if < 1,000 MW
            formatted_results[key] = f"{value:.2f} MWh"

    # Add formatted objective
    formatted_results["Objective"] = f"{objective:,.2f} £"

    return formatted_results



def format_value(value, unit_type="energy"):
    """
    Format numerical values based on unit type.
    
    Parameters:
    - value (float): The numerical value to format.
    - unit_type (str): Type of unit ("energy", "currency", or "emissions").
    
    Returns:
    - str: Formatted string with the appropriate unit.
    """
    if unit_type == "energy":  # Convert MW to GWh
        return f"{value / 1e3:.2f} GWh"
    elif unit_type == "currency":  # Format as GBP (£)
        return f"{value:,.2f} £"
    elif unit_type == "emissions":  # Format CO2 emissions in tonnes
        return f"{value:.2f} tonnes"
    else:
        return f"{value:.2f}"
    



def save_simulation_results(model, genCapacities, multinet, cost_results, output_folder="Output"):
    """
    Saves simulation results, including demand, generation, costs, and emissions, into an Excel file
    and prints them in a formatted way.
    """
    import os
    import pandas as pd

    os.makedirs(output_folder, exist_ok=True)

    total_demand = model.Q_e
    total_gen_gtm = genCapacities[0].sum()
    total_gen_opf = sum(multinet['nets']['power']['res_gen']['p_mw'])

    # Extract cost and emissions from the dictionary
    cost_GT = cost_results["cost_GT"]
    cost_OPGF = cost_results["cost_OPGF"]
    emission_GT = cost_results["emission_GT"]
    co2_cost_GT = cost_results["co2_cost_GT"]
    emission_OPGF = cost_results["emission_OPGF"]
    co2_cost_OPGF = cost_results["co2_cost_OPGF"]

    # Print formatted results
    print("\nSimulation Results:")
    print(f"Total Demand: {format_value(total_demand, 'energy')}")
    print(f"Total Generation GTM: {format_value(total_gen_gtm, 'energy')}")
    print(f"Total Generation OPF: {format_value(total_gen_opf, 'energy')}")
    
    print(f"Total Cost GT: {format_value(cost_GT, 'currency')}")
    print(f"Total Cost OPGF: {format_value(cost_OPGF, 'currency')}")
    print(f"Cost Difference (GT - OPGF): {format_value(cost_GT - cost_OPGF, 'currency')}")
    
    print(f"CO2 Cost (GT Model): {format_value(co2_cost_GT, 'currency')}")
    print(f"CO2 Cost (OPGF Model): {format_value(co2_cost_OPGF, 'currency')}")
    
    print(f"CO2 Emissions (GT Model): {format_value(emission_GT, 'emissions')}")
    print(f"CO2 Emissions (OPGF Model): {format_value(emission_OPGF, 'emissions')}")

    # Organizing results into a dictionary
    results = {
        "Metric": [
            "Total Demand",
            "Total Generation",
            "Total Cost",
            "CO2 Cost",
            "CO2 Emissions"
        ],
        "GT Model": [
            total_demand,
            total_gen_gtm,
            cost_GT,
            co2_cost_GT,
            emission_GT
        ],
        "OPGF Model": [
            total_demand,
            total_gen_opf,
            cost_OPGF,
            co2_cost_OPGF,
            emission_OPGF
        ],
        "Cost/Emission Difference (GT - OPGF)": [
            0,
            total_gen_gtm - total_gen_opf,
            cost_GT - cost_OPGF,
            co2_cost_GT - co2_cost_OPGF,
            emission_GT - emission_OPGF
        ]
    }

    df_results = pd.DataFrame(results)
    output_path = os.path.join(output_folder, "Simulation_results.xlsx")
    df_results.to_excel(output_path, index=False)

    # print(f"\nResults saved to: {output_path}")
    
    return output_path



def save_opgf_generations(multinet, systemData, output_folder="Output"):

    os.makedirs(output_folder, exist_ok=True)

    # OPGF Generation Results
    opgf_gens = pd.DataFrame({
        'Generation (MW)': multinet.nets['power'].res_gen['p_mw'].copy(),
        'Type': systemData['Players']['type']
    })

    opgf_gens_grouped = opgf_gens.groupby('Type', as_index=False).sum()
    opgf_gens_grouped = opgf_gens_grouped.sort_values(by="Generation (MW)", ascending=False)  # Sort

    # Save results to an Excel file
    output_path = os.path.join(output_folder, "OPGF_generation_results.xlsx")
    opgf_gens_grouped.to_excel(output_path, sheet_name="OPGF Model", index=False)

    # Create and save a sorted bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(opgf_gens_grouped["Type"], opgf_gens_grouped["Generation (MW)"], color='blue', zorder=2)
    plt.xlabel("Generation Type")
    plt.ylabel("Generation (MW)")
    plt.title("OPGF Model Generation")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)  # Background grid

    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_folder, "Fig_OPGF_only_gens.png")
    plt.savefig(fig_path, dpi=360)
    # plt.close()




def save_generation_results(model, selected_players, multinet, systemData, output_folder="Output"):


    os.makedirs(output_folder, exist_ok=True)

    # OPGF Generation Results
    opgf_gens = pd.DataFrame({
        'Generation (MW)': multinet.nets['power'].res_gen['p_mw'].copy(),
        'Type': systemData['Players']['type']
    })

    opgf_gens_grouped = opgf_gens.groupby('Type', as_index=False).sum()
    opgf_gens_grouped = opgf_gens_grouped.sort_values(by="Generation (MW)", ascending=False)  # Sort

    # GT Generation Results
    genCapacities = model.q.extract_values()
    gt_gens = pd.DataFrame.from_dict(genCapacities, orient='index', columns=['Generation (MW)'])
    gt_gens['Type'] = systemData['Players']['type'][selected_players].values

    gt_gens_grouped = gt_gens.groupby('Type', as_index=False).sum()
    gt_gens_grouped = gt_gens_grouped.sort_values(by="Generation (MW)", ascending=False)  # Sort

    # Save results to an Excel file with two sheets
    output_path = os.path.join(output_folder, "Generation_results.xlsx")
    with pd.ExcelWriter(output_path) as writer:
        gt_gens_grouped.to_excel(writer, sheet_name="GT Model", index=False)
        opgf_gens_grouped.to_excel(writer, sheet_name="OPGF Model", index=False)

    # Create and save a sorted bar plot for OPGF
    plt.figure(figsize=(12, 6))
    plt.bar(opgf_gens_grouped["Type"], opgf_gens_grouped["Generation (MW)"], color='blue', zorder=2)
    plt.xlabel("Generation Type")
    plt.ylabel("Generation (MW)")
    plt.title("OPGF Model Generation")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)  # Background grid

    plt.tight_layout()

    # Save figure
    fig_opgf_path = os.path.join(output_folder, "Fig_main_OPGF_gens.png")
    plt.savefig(fig_opgf_path, dpi=360)
    # plt.close()

    # Create and save a sorted bar plot for GT Model
    plt.figure(figsize=(12, 6))
    plt.bar(gt_gens_grouped["Type"], gt_gens_grouped["Generation (MW)"], color='red', zorder=2)
    plt.xlabel("Generation Type")
    plt.ylabel("Generation (MW)")
    plt.title("GT Model Generation")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)  # Background grid

    plt.tight_layout()

    # Save figure
    fig_gt_path = os.path.join(output_folder, "Fig_main_GT_gens.png")
    plt.savefig(fig_gt_path, dpi=360)
    # plt.close()



