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


def plot_generation_bar(generation_by_type, output_folder="Output"):

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder)

    # Plot settings
    plt.figure(figsize=(12, 6))
    generation_by_type.sort_values(ascending=False).plot(kind='bar', color='royalblue', edgecolor='black')

    # Labels and title
    plt.xlabel("Generation Type")
    plt.ylabel("Power Output (MW)")
    plt.title("Electricity Generation by Type")
    plt.xticks(rotation=45, ha='right')  # Rotate labels for readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the figure
    plt.savefig(output_path+'/'+'Fig_generation_plot.png', dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to free memory

