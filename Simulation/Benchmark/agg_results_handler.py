# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 19:40:01 2025

@author: Mhdella
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic  # For calculating distances between coordinates
from openpyxl import load_workbook
import shutil
from matplotlib.image import imread



# Coordinates for the regions (17 regions) with acronyms
regions = {
    "NS": (57.5000, -4.0000),  
    "SS": (55.8667, -3.5000),  
    "NWE": (53.7500, -2.5000), 
    "NEE": (54.9000, -1.5000), 
    "NW": (53.3000, -3.0000), 
    "Y": (53.9000, -1.4000),  
    "SW": (51.8333, -3.2500), 
    "WM": (52.5000, -1.9167), 
    "EM": (52.9500, -0.9500),  
    "SWE": (50.8000, -3.5000), 
    "SE": (51.0000, -1.0000),  
    "L": (51.5099, -0.1181), 
    "EE": (52.3000, 0.7000), 
    "SEE": (51.2000, 0.7000),  
    "IE": (53.5000, -7.0000), 
    "CE": (50.5000, 8.0000),  
    "NS_OFF": (57.5000, -1.950), 
}


# Hydrogen network connections with capacity
connections = [
    ("NS", "SS", 660),
    ("SS", "NWE", 10182),
    ("SS", "NEE", 2413),
    ("NWE", "NEE", 216),
    ("NWE", "NW", 1020),
    ("NWE", "Y", 2086),
    ("NWE", "WM", 7021),
    ("NEE", "Y", 98),
    ("Y", "EM", 1045),
    ("SW", "SWE", 1903),
    ("WM", "EM", 1708),
    ("WM", "SWE", 3742),
    ("EM", "EE", 26),
    ("SWE", "SE", 393),
    ("SE", "EE", 4639),
    ("L", "EE", 8916),
    ("L", "SEE", 4622),
    ("NS", "NEE", 2081),
]



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
    



def save_simulation_results(model, genCapacities, multinet, cost_results, output_folder):
    """
    Saves simulation results, including demand, generation, costs, and emissions, into an Excel file
    and prints them in a formatted way.
    """
    import os
    import pandas as pd
    
    # Constants
    energy_hydrogen = 39.41  # kWh/kg, energy content of hydrogen
    energy_gas = 14.64 # kWh/kg, energy content of natural gas


    os.makedirs(output_folder, exist_ok=True)

    Ele_demand = sum(multinet.nets['power'].load['p_mw'])
    H2_demand = multinet.nets['hydrogen'].res_sink['mdot_kg_per_s'].sum(skipna=True)*(energy_hydrogen*3600)/1e3
    total_demand = Ele_demand + H2_demand

    total_gen_gtm = genCapacities[0].sum()
    

    total_gen_gtm = genCapacities[0].sum()

    # total_gen_opf = sum(multinet['nets']['power']['res_gen']['p_mw'])
    gen_opf_wo_g2p = sum(multinet['nets']['power']['res_gen']['p_mw'])
    gen_g2p = sum(multinet.nets['power'].res_sgen['p_mw'])
    # total_gen_opf = gen_opf_wo_g2p + gen_g2p 
    total_gen_opf = gen_opf_wo_g2p + gen_g2p + H2_demand

    # Extract cost and emissions from the dictionary
    cost_GT = cost_results["cost_GT"]
    cost_OPGF = cost_results["cost_OPGF"]
    emission_GT = cost_results["emission_GT"]
    co2_cost_GT = cost_results["co2_cost_GT"]
    emission_OPGF = cost_results["emission_OPGF"]
    co2_cost_OPGF = cost_results["co2_cost_OPGF"]
    

    # # Print formatted results
    # print("\nSimulation Results:")
    # print(f"Total Demand: {format_value(total_demand, 'energy')}")
    # print(f"Electric Demand: {format_value(Ele_demand, 'energy')}")
    # print(f"Hydrogen Demand: {format_value(H2_demand, 'energy')}")

    # print(f"Total Generation GTM: {format_value(total_gen_gtm, 'energy')}")
    # print(f"Total Generation OPGF: {format_value(total_gen_opf, 'energy')}")
    
    # print(f"Total Cost GT: {format_value(cost_GT, 'currency')}")
    # print(f"Total Cost OPGF: {format_value(cost_OPGF, 'currency')}")
    # print(f"Cost Difference (GT - OPGF): {format_value(cost_GT - cost_OPGF, 'currency')}")
    
    # print(f"CO2 Cost (GT Model): {format_value(co2_cost_GT, 'currency')}")
    # print(f"CO2 Cost (OPGF Model): {format_value(co2_cost_OPGF, 'currency')}")
    
    # print(f"CO2 Emissions (GT Model): {format_value(emission_GT, 'emissions')}")
    # print(f"CO2 Emissions (OPGF Model): {format_value(emission_OPGF, 'emissions')}")


    # Organizing results into a dictionary
    results = {
        "Metric": [
            "Total Demand",
            "Electric Demand",
            "Hydrogen Demand",
            "Total Generation",
            "Total Cost",
            "CO2 Cost",
            "CO2 Emissions"
        ],
        "GT Model": [
            total_demand,
            Ele_demand,
            H2_demand,
            total_gen_gtm,
            cost_GT,
            co2_cost_GT,
            emission_GT
        ],
        "OPGF Model": [
            total_demand,
            Ele_demand,
            H2_demand,
            total_gen_opf,
            cost_OPGF,
            co2_cost_OPGF,
            emission_OPGF
        ],
        "Cost/Emission Difference (GT - OPGF)": [
            0,
            0,
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




def save_opgf_generations(multinet, systemData, output_folder):
    # Mapping for the labels to improve clarity
    label_mapping = {
        'g2p(H2-CCGT)': 'P-to-G (H2-CCGT)',
        'g2p(H2-OCGT)': 'P-to-G (H2-OCGT)',
        'g2p(Fuel Cell)': 'P-to-G (Fuel Cell)',
        'p2g': 'P-to-G',
        'g2g': 'G-to-G'
    }

    # Apply the label mapping to the 'type' column in systemData
    systemData['Players']['type'] = systemData['Players']['type'].replace(label_mapping)

    os.makedirs(output_folder, exist_ok=True)

    # OPGF Generation Results
    opgf_gens = pd.DataFrame({
        'Generation (MW)': multinet.nets['power'].res_gen['p_mw'].copy(),
        'Type': systemData['Players']['type']
    })

    opgf_gens_grouped = opgf_gens.groupby('Type', as_index=False).sum()

    # Add H2-Storage to grouped generation data
    hydrogen_related_types = ['P-to-G', 'G-to-G', 'P-to-G (Fuel Cell)', 'P-to-G (H2-CCGT)', 'P-to-G (H2-OCGT)']
    hydrogen_related_types.append('H2-Storage')  # Add H2-Storage here

    # Extract the H2-Storage generation and append it to the grouped DataFrame
    hydrogen_gens = systemData['Agg_Players'][systemData['Agg_Players']['type'].isin(hydrogen_related_types)][['max_p_mw', 'type']].copy()
    hydrogen_gens.columns = ['Generation (MW)', 'Type']

    # Append H2-Storage data
    opgf_gens_grouped = pd.concat([opgf_gens_grouped, hydrogen_gens], ignore_index=True)

    # Sort by Generation (MW)
    opgf_gens_grouped = opgf_gens_grouped.sort_values(by="Generation (MW)", ascending=False)

    # Save results to an Excel file
    output_path = os.path.join(output_folder, "OPGF_generation_results.xlsx")
    opgf_gens_grouped.to_excel(output_path, sheet_name="OPGF Model", index=False)

    # Create and save a sorted bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(opgf_gens_grouped["Type"], opgf_gens_grouped["Generation (MW)"] / 1000, color='blue', zorder=2)  # Convert MW to GW
    plt.xlabel("Generation Type", fontsize=14)
    plt.ylabel("Generation (GW)", fontsize=14)
    plt.title("OPGF Model Generation", fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_folder, "Fig_OPGF_only_gens.png")
    plt.savefig(fig_path, dpi=360)
    # plt.close()




def save_generation_results(model, selected_players, multinet, systemData, output_folder):
    
    
    
    # Mapping for the labels to improve clarity
    label_mapping = {
        'g2p(H2-CCGT)': 'G2P (H2-CCGT)',
        'g2p(H2-OCGT)': 'G2P (H2-OCGT)',
        'g2p(Fuel Cell)': 'G2P (Fuel Cell)',
        'p2g': 'P2G',
        'g2g': 'G2G+CCS',
        'Storage': 'BESS'
    }

    # Apply the label mapping
    systemData['Agg_Players']['type'] = systemData['Agg_Players']['type'].replace(label_mapping)

    systemData['Agg_Players'] = systemData['Agg_Players'].drop_duplicates(subset=['type'], keep='last')

    
    os.makedirs(output_folder, exist_ok=True)

    # OPGF Generation Results
    opgf_gens = pd.DataFrame({
        'Generation (MW)': multinet.nets['power'].res_gen['p_mw'].copy(),
        'Type': systemData['Players']['type']
    })
    opgf_gens_grouped = opgf_gens.groupby('Type', as_index=False).sum()
    opgf_gens_grouped['Type'] = opgf_gens_grouped['Type'].replace(label_mapping)

    # GT Generation Results
    genCapacities = model.q.extract_values()
    gt_gens = pd.DataFrame.from_dict(genCapacities, orient='index', columns=['Generation (MW)'])
    gt_gens['Type'] = systemData['Agg_Players']['type'][selected_players].values
    gt_gens_grouped = gt_gens.groupby('Type', as_index=False).sum()

    # Hydrogen-related types
    hydrogen_related_types = ['P2G', 'G2G+CCS', 'G2P (Fuel Cell)', 'G2P (H2-CCGT)', 'G2P (H2-OCGT)', 'H2-Storage']
    update_agg_players_max_p_mw(multinet, systemData, selected_players)
    hydrogen_gens = systemData['Agg_Players'][systemData['Agg_Players']['type'].isin(hydrogen_related_types)][['max_p_mw', 'type']].copy()
    hydrogen_gens.columns = ['Generation (MW)', 'Type']
    opgf_gens_grouped = pd.concat([opgf_gens_grouped, hydrogen_gens], ignore_index=True)

    def add_g_to_p_summary(df):
        g_to_p_sum_g2p = df[df['Type'].isin(['G2P (Fuel Cell)', 'G2P (H2-CCGT)', 'G2P (H2-OCGT)'])]['Generation (MW)'].sum()
        g_to_p_sum_wind = df[df['Type'].isin(['Offshore Wind', 'Onshore Wind'])]['Generation (MW)'].sum()
        g_to_p_sum_hydro = df[df['Type'].isin(['Hydro ROR', 'Hydro reservoir'])]['Generation (MW)'].sum()
        g_to_p_sum_chp = df[df['Type'].isin(['Micro CHP', 'Industrial CHP'])]['Generation (MW)'].sum()

        df = pd.concat([df, pd.DataFrame({'Type': ['G2P'], 'Generation (MW)': [g_to_p_sum_g2p]})], ignore_index=True)
        df = pd.concat([df, pd.DataFrame({'Type': ['Wind'], 'Generation (MW)': [g_to_p_sum_wind]})], ignore_index=True)
        df = pd.concat([df, pd.DataFrame({'Type': ['Hydropower'], 'Generation (MW)': [g_to_p_sum_hydro]})], ignore_index=True)
        df = pd.concat([df, pd.DataFrame({'Type': ['CHP'], 'Generation (MW)': [g_to_p_sum_chp]})], ignore_index=True)

        df = df[~df['Type'].isin(['G2P (Fuel Cell)', 'G2P (H2-CCGT)', 'G2P (H2-OCGT)',
                                  'Offshore Wind', 'Onshore Wind',
                                  'Hydro ROR', 'Hydro reservoir',
                                  'Micro CHP', 'Industrial CHP'])]
        
        df = df.drop_duplicates(subset=['Type'], keep='last')
        
        
        return df
    

    # Update both dataframes
    opgf_gens_grouped = add_g_to_p_summary(opgf_gens_grouped)
    gt_gens_grouped = add_g_to_p_summary(gt_gens_grouped)

    # Add percentage column
    for df in [opgf_gens_grouped, gt_gens_grouped]:
        total_gen = abs(df['Generation (MW)']).sum()
        df['Percentage (%)'] = (abs(df['Generation (MW)']) / total_gen * 100).round(2).astype(str) + "%"


    # Sorting
    opgf_gens_grouped['Generation (MW)'] = opgf_gens_grouped['Generation (MW)'].abs()
    opgf_gens_grouped = opgf_gens_grouped.sort_values(by="Generation (MW)", ascending=False)
    gt_gens_grouped = gt_gens_grouped.sort_values(by="Generation (MW)", ascending=False)

    
    # Save results to Excel
    output_path = os.path.join(output_folder, "Generation_results.xlsx")
    with pd.ExcelWriter(output_path) as writer:
        gt_gens_grouped.to_excel(writer, sheet_name="GT Model", index=False)
        opgf_gens_grouped.to_excel(writer, sheet_name="OPGF Model", index=False)

    systemData['gt_mix'] = gt_gens_grouped
    systemData['opgf_mix'] = opgf_gens_grouped

    def plot_generation(data, title, color, output_path):
        threshold = 1e-3
        data_filtered = data[abs(data["Generation (MW)"]) >= threshold].copy()

        plt.figure(figsize=(12, 6))
        bars = plt.bar(data_filtered["Type"], data_filtered["Generation (MW)"] / 1000, color=color, zorder=2)
        plt.ylabel("Energy (GW)", fontsize=18)
        plt.xticks(rotation=45, ha="right", fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
        plt.tight_layout()

        # Add percentage labels above bars
        for bar, perc in zip(bars, data_filtered["Percentage (%)"]):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height, perc,
                     ha='center', va='bottom', fontsize=12, rotation=0)

        plt.savefig(output_path, dpi=360)

    # Plot results
    plot_generation(opgf_gens_grouped, "OPGF Model Generation", 'blue', os.path.join(output_folder, "Fig_main_OPGF_gens.png"))
    plot_generation(gt_gens_grouped, "GT Model Generation", 'red', os.path.join(output_folder, "Fig_main_GT_gens.png"))

    
    
    # At the end of save_generation_results()
    npv_results = NPV_Calculation(model, systemData, output_path, sheet_name="OPGF Model")




def get_storage_generation(multinet, systemData):

    generation = multinet.nets['power'].res_gen[['p_mw']].copy()
    
    players_type = systemData['Players'][['type', 'id']].copy()
    generation = generation.merge(players_type, left_index=True, right_on='id')
    
    storage_generation = generation[generation['type'] == 'Storage']['p_mw'].sum()
    
    return storage_generation



# Function to calculate distance between two coordinates
def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).kilometers

# Convert MWh to mdot
def convert_mwh_to_mdot(mwh, energy_hydrogen):
    """Convert TWh to mdot_kg_per_s using energy content of hydrogen."""
    return (mwh * 1e3) / (energy_hydrogen * 3600)



def OPGF_results(multinet, systemData):
        
    # Constants
    energy_hydrogen = 39.41  # kWh/kg, energy content of hydrogen
    energy_gas = 14.64 # kWh/kg, energy content of natural gas


    results = {        
        "Maximum generation capacity": sum(multinet.nets['power'].gen['max_p_mw']),
        "Total Generation": sum(multinet.nets['power'].res_gen['p_mw'])+sum(multinet.nets['power'].res_sgen['p_mw']) + abs(get_storage_generation(multinet, systemData)),
        "G2P Generation": sum(multinet.nets['power'].res_sgen['p_mw']),
        "Other Generations": sum(multinet.nets['power'].res_gen['p_mw']),
        "Storage Generation": get_storage_generation(multinet, systemData),
    
        "Total Load": sum(multinet.nets['power'].res_load['p_mw']) + abs(get_storage_generation(multinet, systemData)),
        
        "Losses": sum(multinet.nets['power'].res_line['pl_mw']),
        "Elec load": sum(multinet.nets['power'].res_load['p_mw'][:14]),
        "Storage Load": - get_storage_generation(multinet, systemData),
        "P2G load": sum(multinet.nets['power'].res_load['p_mw'][14:65]),
        "CCS load": sum(multinet.nets['power'].res_load['p_mw'][65:]),
        
        "External grid": multinet.nets['power'].res_ext_grid['p_mw'][0],
        "G2G Gas Demand": multinet.nets['gas'].res_sink['mdot_kg_per_s'][:34].sum(skipna=True)*(energy_gas*3600)/1e3,
    
        "Total H2 Demand": multinet.nets['hydrogen'].res_sink['mdot_kg_per_s'].sum(skipna=True)*(energy_hydrogen*3600)/1e3,
        "H2 Consumption": multinet.nets['hydrogen'].res_sink['mdot_kg_per_s'][:14].sum(skipna=True)*(energy_hydrogen*3600)/1e3,
        "H2 G2P Demand": multinet.nets['hydrogen'].res_sink['mdot_kg_per_s'][14:].sum(skipna=True)*(energy_hydrogen*3600)/1e3,
    
        "Total H2 Supply": multinet.nets['hydrogen'].res_source['mdot_kg_per_s'].sum(skipna=True)*(energy_hydrogen*3600)/1e3,
        "P2G H2 Supply": multinet.nets['hydrogen'].res_source['mdot_kg_per_s'][0:51].sum(skipna=True)*(energy_hydrogen*3600)/1e3,
        "G2G H2 Supply":multinet.nets['hydrogen'].res_source['mdot_kg_per_s'][51:].sum(skipna=True)*(energy_hydrogen*3600)/1e3,
        "H2 external": sum(multinet.nets['hydrogen'].res_ext_grid['mdot_kg_per_s'])*(energy_hydrogen*3600)/1e3
    }
    
    return results



def plot_capex_vs_output(model, systemData, output_folder):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    # Extract data
    players = list(model.p)
    q_values = [model.q[p].value for p in players if model.q[p].value is not None]
    fc_values = [model.eps[p] for p in players if model.q[p].value is not None]
    player_ids = [p for p in players if model.q[p].value is not None]
    player_names = [systemData['Agg_Players']['type'].get(p, str(p)) for p in player_ids]

    # Custom color map by technology
    custom_colors = {
        'PV': 'orange',
        'Offshore Wind': 'darkblue',
        'Onshore Wind': 'skyblue',
        'Biomass': 'saddlebrown',
        'Gas CCS': 'gray',
        'Nuclear': 'purple',
        'Hydro ROR': 'mediumblue',
        'Hydro reservoir': 'steelblue',
        'Geothermal': 'firebrick',
        'BESS': 'gold',
        'Micro CHP': 'darkgreen',
        'Industrial CHP': 'green',
        'G2P (H2-CCGT)': 'cyan',
        'G2P (H2-OCGT)': 'deepskyblue',
        'G2P (Fuel Cell)': 'turquoise',
        'P2G': 'lightseagreen',
        'G2G+CCS': 'darkblue',
        'Other RES': 'olive'
    }

    # Optional: define consistent markers
    markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X', '<', '>', 'H', '8', 'p', 'h']
    unique_names = list(dict.fromkeys(player_names))  # preserve order

    style_map = {
        name: (custom_colors.get(name, 'black'), markers[i % len(markers)])
        for i, name in enumerate(unique_names)
    }


    marker_size = 100
    
    # Plot
    plt.figure(figsize=(12, 7))
    for i, name in enumerate(player_names):
        color, marker = style_map[name]
        plt.scatter(q_values[i], fc_values[i], color=color, marker=marker, s=marker_size, label=name)
        

    # Create legend without duplicates
    handles_labels = {}
    for name in unique_names:
        color, marker = style_map[name]
        handles_labels[name] = plt.Line2D(
            [0], [0], color=color, marker=marker,
            linestyle='', markersize=8, label=name
        )

    plt.legend(
        handles=handles_labels.values(),
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        title="Player Types"
    )

    plt.xlabel("Energy Output, Peak Demand (MWh)")
    plt.ylabel("CAPEX in (m£)")
    plt.title("CAPEX vs. Energy Output for Different Energy Assets")
    plt.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Room for legend
    
    # Save the plot
    plot_filename = os.path.join(output_folder, "capex_vs_output.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    
    
def update_gt_with_opgf_results(model, multinet, systemData, selected_players):

    # OPGF Generation Results
    opgf_gens = pd.DataFrame({
        'Generation (MW)': multinet.nets['power'].res_gen['p_mw'].copy(),
        'Type': systemData['Players']['type']
    })

    # Group OPGF generation by type
    opgf_gens_grouped = opgf_gens.groupby('Type', as_index=False).sum()

    # Extract GT generation from model.q
    gt_gens = pd.DataFrame.from_dict(
        model.q.extract_values(), orient='index', columns=['Generation (MW)']
    )
    gt_gens['Type'] = systemData['Agg_Players']['type'][selected_players].values

    hydrogen_related_types = ['p2g', 'g2g', 'g2p(Fuel Cell)', 'g2p(H2-CCGT)', 'g2p(H2-OCGT)']
    
    # # Filter the rows from gt_gens_grouped that belong to hydrogen-related types
    # hydrogen_gens = gt_gens[gt_gens['Type'].isin(hydrogen_related_types)]
    
    update_agg_players_max_p_mw(multinet, systemData, selected_players)
    
    hydrogen_gens = systemData['Agg_Players'][systemData['Agg_Players']['type'].isin(hydrogen_related_types)][['max_p_mw', 'type']].copy()
    hydrogen_gens.columns = ['Generation (MW)', 'Type']
    
    opgf_gens_grouped = pd.concat([opgf_gens_grouped, hydrogen_gens], ignore_index=True)

    # Merge and preserve index
    merged_df = gt_gens.merge(
        opgf_gens_grouped, on="Type", how="left", suffixes=('_gt', '_opgf')
    )
    merged_df.index = gt_gens.index

    # Assign to model.q with safety for non-negativity
    for idx, val in merged_df['Generation (MW)_opgf'].items():
        model.q[idx] = max(0, val)
        
    return model



def update_agg_players_max_p_mw(multinet, systemData, selected_players):

    resulted_gens = multinet['nets']['power']['res_gen']['p_mw']
    players_data = systemData['Players'].copy()  
    players_data['updated_max_p_mw'] = resulted_gens.values  # Add the results
    aggregated_updates = players_data.groupby('type')['updated_max_p_mw'].sum()
    agg_players_data = systemData['Agg_Players'].copy()  
    agg_players_data['max_p_mw'] = agg_players_data['type'].map(aggregated_updates).fillna(agg_players_data['max_p_mw'])
    hydrogen_types = ['g2p(H2-CCGT)', 'g2p(H2-OCGT)', 'g2p(Fuel Cell)', 'p2g', 'g2g']

    from agg_results_handler import OPGF_results
    opf_results = OPGF_results(multinet, systemData)

    hydrogen_updates = {
        'p2g': opf_results['P2G H2 Supply'],  # max_p_mw for 'p2g'
        'g2g': opf_results['G2G H2 Supply'],  # max_p_mw for 'g2g'
        'g2p': opf_results['G2P Generation'],  # max_p_mw for 'g2p'
        'g2p(H2-OCGT)': opf_results['G2P Generation'] / 3,  # max_p_mw for 'g2p(H2-OCGT)'
        'g2p(Fuel Cell)': opf_results['G2P Generation'] / 3,  # max_p_mw for 'g2p(Fuel Cell)'
        'g2p(H2-CCGT)': opf_results['G2P Generation'] / 3   # max_p_mw for 'g2p(H2-CCGT)'
    }
    

    for h_type, updated_value in hydrogen_updates.items():
        if h_type in agg_players_data['type'].values:
            agg_players_data.loc[agg_players_data['type'] == h_type, 'max_p_mw'] = updated_value
            
           

    systemData['Agg_Players']['max_p_mw'][selected_players] = agg_players_data.loc[selected_players, 'max_p_mw']


    threshold = 1e-5
    systemData['Agg_Players']['max_p_mw'] = systemData['Agg_Players']['max_p_mw'].apply(lambda x: 0 if abs(x) < threshold else x)
    
    
    if abs(opf_results['H2 external']) > 1e-5:
        systemData['Agg_Players'] = pd.concat([
            systemData['Agg_Players'],
            pd.DataFrame([{
                'max_p_mw': opf_results['H2 external'],
                # 'max_p_mw': abs(opf_results['H2 external']),
                'type': 'H2-Storage',
                'costs': 0, 'epsilon': 0, 'phi': 0,
                'economic_life': 0, 'emissions': 0,
                'discount_rate': 0, 'CF': 0,
                'opex_mw_year': 0, 'id': systemData['Agg_Players']['id'].max() + 1,
                'zone_id': 0, 'scale': 1, 'Efficiency': 1
            }])], ignore_index=True)

    return systemData['Agg_Players']






def plot_convergence(output_folder):
    import matplotlib.pyplot as plt
    df = pd.read_csv(os.path.join(output_folder, 'convergence.csv'))
    plt.figure(figsize=(10, 5))
    plt.plot(df['iteration'], df['gen_diff'], label='Generation Difference')
    plt.plot(df['iteration'], df['cost_diff'], label='Cost Difference')
    plt.xlabel('Iteration')
    plt.ylabel('Difference')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'convergence_plot.png'))
    # plt.close()






def save_cost_calc(systemData, output_folder):
    # === Load opex costs ===
    cost_df = systemData['Opex_Costs'].copy()
    tech_costs = cost_df['Opex_Cost (£/MWh)'].to_dict()

    # === Compute averages for grouped types ===
    tech_costs['Wind'] = (tech_costs.get('Offshore Wind', 0) + tech_costs.get('Onshore Wind', 0)) / 2
    tech_costs['Hydropower'] = (tech_costs.get('Hydro ROR', 0) + tech_costs.get('Hydro reservoir', 0)) / 2
    tech_costs['CHP'] = (tech_costs.get('Micro CHP', 0) + tech_costs.get('Industrial CHP', 0)) / 2

    # === Load GT and OPGF mixes ===
    gt_df = systemData['gt_mix'].copy()
    opgf_df = systemData['opgf_mix'].copy()
    # gt_df.columns = ['Type', 'Generation_MWh']
    # opgf_df.columns = ['Type', 'Generation_MWh']
    gt_df.columns = ['Type', 'Generation_MWh', 'Percentage (%)']
    opgf_df.columns = ['Type', 'Generation_MWh', 'Percentage (%)']


    # === Cost calculation ===
    def compute_cost(df, label):
        cost = 0
        missing = []
        for _, row in df.iterrows():
            tech = row['Type']
            gen = row['Generation_MWh']
            if tech in tech_costs:
                cost += gen * tech_costs[tech]
            else:
                missing.append(tech)
        if missing:
            print(f"[{label}] Missing cost entries for: {missing}")
        return cost

    cost_GT = compute_cost(gt_df, "GT")
    cost_OPGF = compute_cost(opgf_df, "OPGF")
    cost_diff = cost_GT - cost_OPGF

    
    # === Load Simulation_results.xlsx and extract OPGF values ===
    sim_path = os.path.join(output_folder, "Simulation_results.xlsx")
    sim_df = pd.read_excel(sim_path)

    # Extract relevant rows from OPGF column
    opgf_metrics = sim_df.set_index('Metric').loc[
        ['Total Demand', 'Electric Demand', 'Hydrogen Demand', 'Total Generation', 'CO2 Emissions'],
        'OPGF Model'
    ]

    # Add the updated cost from our calculation
    opgf_metrics['Total Cost'] = cost_OPGF

    # # Reorder and reset index
    final_df = opgf_metrics[['Total Demand', 'Electric Demand', 'Hydrogen Demand', 'Total Generation', 'Total Cost', 'CO2 Emissions']].T
    final_df = final_df.to_frame(name='Value').reset_index()
    final_df.columns = ['Metric', 'Value']
    

    # === Save to Simulation_results.xlsx as second sheet ===
    with pd.ExcelWriter(sim_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        final_df.to_excel(writer, sheet_name="Simulation_results", index=False)
        
        
    # === Save results to Excel ===
    result_df = pd.DataFrame({
        'Metric': ['Total Cost (GT)', 'Total Cost (OPGF)', 'Cost Difference (GT - OPGF)'],
        'Value': [cost_GT, cost_OPGF, cost_diff]
    })

    output_path = os.path.join(output_folder, "Generation_results.xlsx")
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        result_df.to_excel(writer, sheet_name="Cost Comparison", index=False)



    return {
        'cost_GT': cost_GT,
        'cost_OPGF': cost_OPGF,
        'cost_diff': cost_diff
    }





def print_simulation_results(systemData, genCapacities, multinet, cost_results, output_folder):
    """
    Saves simulation results, including demand, generation, costs, and emissions, into an Excel file
    and prints them in a formatted way.
    """
    import os
    import pandas as pd
    
    # Constants
    energy_hydrogen = 39.41  # kWh/kg, energy content of hydrogen
    energy_gas = 14.64 # kWh/kg, energy content of natural gas


    os.makedirs(output_folder, exist_ok=True)

    Ele_demand = sum(multinet.nets['power'].load['p_mw'])
    H2_demand = multinet.nets['hydrogen'].res_sink['mdot_kg_per_s'].sum(skipna=True)*(energy_hydrogen*3600)/1e3
    total_demand = Ele_demand + H2_demand

    total_gen_gtm = genCapacities[0].sum()
    

    total_gen_gtm = genCapacities[0].sum()

    # total_gen_opf = sum(multinet['nets']['power']['res_gen']['p_mw'])
    gen_opf_wo_g2p = sum(multinet['nets']['power']['res_gen']['p_mw'])
    gen_g2p = sum(multinet.nets['power'].res_sgen['p_mw'])
    # total_gen_opf = gen_opf_wo_g2p + gen_g2p 
    total_gen_opf = gen_opf_wo_g2p + gen_g2p + H2_demand

    # Extract cost and emissions from the dictionary
    op_costs2 = save_cost_calc(systemData, output_folder)
    cost_GT = op_costs2["cost_GT"]
    cost_OPGF = op_costs2["cost_OPGF"]
    
    emission_GT = cost_results["emission_GT"]
    co2_cost_GT = cost_results["co2_cost_GT"]
    emission_OPGF = cost_results["emission_OPGF"]
    co2_cost_OPGF = cost_results["co2_cost_OPGF"]
    

    # Print formatted results
    print("\nSimulation Results:")
    print(f"Total Demand: {format_value(total_demand, 'energy')}")
    print(f"Electric Demand: {format_value(Ele_demand, 'energy')}")
    print(f"Hydrogen Demand: {format_value(H2_demand, 'energy')}")

    print(f"Total Generation GTM: {format_value(total_gen_gtm, 'energy')}")
    print(f"Total Generation OPGF: {format_value(total_gen_opf, 'energy')}")
    
    print(f"Total Cost GT: {format_value(cost_GT, 'currency')}")
    print(f"Total Cost OPGF: {format_value(cost_OPGF, 'currency')}")
    print(f"Cost Difference (GT - OPGF): {format_value(cost_GT - cost_OPGF, 'currency')}")
    
    print(f"CO2 Cost (GT Model): {format_value(co2_cost_GT, 'currency')}")
    print(f"CO2 Cost (OPGF Model): {format_value(co2_cost_OPGF, 'currency')}")
    
    print(f"CO2 Emissions (GT Model): {format_value(emission_GT, 'emissions')}")
    print(f"CO2 Emissions (OPGF Model): {format_value(emission_OPGF, 'emissions')}")

    
    return 




def process_policy_support_and_save(model, multinet, systemData, df_shap, output_folder,
                                     pos_shap_elec, pos_shap_hydrogen,
                                     neg_shap_elec, neg_shap_hydrogen,
                                     calculate_policy_opex_support,
                                     calculate_policy_npv_support):
    
    # Constants
    energy_hydrogen = 39.41  # kWh/kg, energy content of hydrogen
    energy_gas = 14.64 # kWh/kg, energy content of natural gas
    
    
    Ele_demand = sum(multinet.nets['power'].load['p_mw'])
    H2_demand = multinet.nets['hydrogen'].res_sink['mdot_kg_per_s'].sum(skipna=True)*(energy_hydrogen*3600)/1e3
    total_demand = Ele_demand + H2_demand

    
    # # === Calculate unmet demand support ===    
    policy_support_e_demand = 0 if not neg_shap_elec else max(0, Ele_demand - sum(model.q[p].value for p in pos_shap_elec))/1000
    policy_support_h_demand = 0 if not neg_shap_hydrogen else max(0, H2_demand - sum(model.q[p].value for p in pos_shap_hydrogen))/1000

    # # === Calculate Opex-based support ===
    # policy_support_electric = calculate_policy_opex_support(model, systemData, neg_shap_elec, demand_type='electric')
    # policy_support_hydrogen = calculate_policy_opex_support(model, systemData, neg_shap_hydrogen, demand_type='hydrogen')

    # === Calculate Opex-based support ===
    policy_support_electric = policy_support_e_demand *model.pE.value*1000/1e6
    policy_support_hydrogen = policy_support_h_demand *model.pH.value*1000/1e6

    # === Calculate NPV-based support ===
    policy_support_npv_electric = calculate_policy_npv_support(model, systemData, neg_shap_elec, demand_type='electric')
    policy_support_npv_hydrogen = calculate_policy_npv_support(model, systemData, neg_shap_hydrogen, demand_type='hydrogen')

    # === Format results ===
    support_data = {
        "Metric": [
            "Support for Electricity Technologies (MW)",
            "Support for Hydrogen Technologies (MW)",
            "Opex-based Policy Support for Electricity Technologies (m£)",
            "Opex-based Policy Support for Hydrogen Technologies (m£)",
            "NPV-based Policy Support for Electricity Technologies (£)",
            "NPV-based Policy Support for Hydrogen Technologies (£)"
        ],
        "Value": [
            round(policy_support_e_demand, 2),
            round(policy_support_h_demand, 2),
            round(policy_support_electric, 2),
            round(policy_support_hydrogen, 2),
            f"{policy_support_npv_electric:.2e}",
            f"{policy_support_npv_hydrogen:.2e}"
        ]
    }

    df_support = pd.DataFrame(support_data)

    # === Sort Shapley DataFrame ===
    df_shap_sorted = df_shap.sort_values(by="Shapley Value", ascending=False)

    # === Save both to Simulation_results.xlsx ===
    sim_path = os.path.join(output_folder, "Simulation_results.xlsx")
    with pd.ExcelWriter(sim_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_support.to_excel(writer, sheet_name="Policy Support", index=False)
        df_shap_sorted.to_excel(writer, sheet_name="Shapley Values", index=False)

    print("\n=== Policy Support ===")
    print(df_support)

    print("\n=== Sorted Shapley Values ===")
    print(df_shap_sorted)




def run_scenario_simulation(phi_val, invest_span, gens_flag, time_steps, demand_level, output_folder):
    import os
    import shutil
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    # Define scenarios and their friendly names
    plan_scenarios = {
        '25GW': "Uniform +25GW/Tech.",
        'HiRES_HiH2': "High RES–High H2",
        'HiRES_LwH2': "High RES–Low H2",
        'LwBESS_HiH2': "High H2–Low BESS"
    }

    # Empty dictionaries to collect results
    simulation_results = {}
    policy_support_results = {}
    npv_results = {}

    for gens_flag, scenario_name in plan_scenarios.items():
        print(f"\nRunning scenario: {scenario_name} ({gens_flag})")

        # Build scenario paths
        main_path = os.path.join(*output_folder.split('\\')[:5])
        last_path = os.path.join(*output_folder.split('\\')[-2:])
        name_folder = 'Case_' + gens_flag
        sim_path = os.path.join(os.path.join(main_path, name_folder, last_path))

        sim_folder = os.path.join(sim_path, "Simulation_results.xlsx")

        # === Copy and collect figures ===
        figures_folder = os.path.join(main_path, "Figures")
        os.makedirs(figures_folder, exist_ok=True)

        fig_src_path = os.path.join(sim_path, "Fig_main_OPGF_gens.png")
        safe_scenario_name = gens_flag.replace('/', '_').replace(' ', '_')
        fig_dest_path = os.path.join(figures_folder, f"Fig_main_OPGF_gens_{safe_scenario_name}.png")

        if os.path.exists(fig_src_path):
            shutil.copy(fig_src_path, fig_dest_path)
            print(f"Copied figure for {scenario_name} to Figures folder.")
        else:
            print(f"Figure not found for {scenario_name}: {fig_src_path}")

        # === Extract results from Simulation_results.xlsx ===
        try:
            df_sim = pd.read_excel(sim_folder, sheet_name="Simulation_results")
            df_policy = pd.read_excel(sim_folder, sheet_name="Policy Support")
        except Exception as e:
            print(f"Failed to read results for scenario {scenario_name}: {e}")
            continue

        # === Save relevant simulation results ===
        simulation_results[scenario_name] = {
            "Total Demand [GWh]": df_sim.loc[df_sim["Metric"] == "Total Demand", "Value"].values[0],
            "Electric Demand [GWh]": df_sim.loc[df_sim["Metric"] == "Electric Demand", "Value"].values[0],
            "Hydrogen Demand [GWh]": df_sim.loc[df_sim["Metric"] == "Hydrogen Demand", "Value"].values[0],
            "Total Energy Supply [GWh]": df_sim.loc[df_sim["Metric"] == "Total Generation", "Value"].values[0],
            "Total Operational Cost [m£]": df_sim.loc[df_sim["Metric"] == "Total Cost", "Value"].values[0],
            "Total CO2 Emissions [Tonnes]": df_sim.loc[df_sim["Metric"] == "CO2 Emissions", "Value"].values[0],
        }

        # === Save policy support results ===
        policy_support_results[scenario_name] = {
            "Support for Electricity [GWh]": df_policy.loc[0, "Value"],
            "Support for Hydrogen [GWh]": df_policy.loc[1, "Value"],
            "Opex-based Support for Electricity [m£]": df_policy.loc[2, "Value"],
            "Opex-based Support for Hydrogen [m£]": df_policy.loc[3, "Value"],
            "NPV-based Support for Electricity [£]": df_policy.loc[4, "Value"],
            "NPV-based Support for Hydrogen [£]": df_policy.loc[5, "Value"],
        }

        # === Extract NPV from Generation_results_with_NPV.xlsx ===
        npv_file = os.path.join(sim_path, "Generation_results_with_NPV.xlsx")
        try:
            df_npv = pd.read_excel(npv_file, sheet_name="Total NPV Summary")
            npv_results[scenario_name] = df_npv.iloc[0]["Total System NPV (£, billion)"]
        except Exception as e:
            print(f"Failed to read NPV for scenario {scenario_name}: {e}")
            npv_results[scenario_name] = None

    # === Create pivoted DataFrames for simulation and policy ===
    df_sim_all = pd.DataFrame(simulation_results).T
    df_sim_all.index.name = 'Scenario'
    df_sim_all = df_sim_all.T.reset_index().rename(columns={"index": "Metric"})

    df_policy_all = pd.DataFrame(policy_support_results).T
    df_policy_all.index.name = 'Scenario'
    df_policy_all = df_policy_all.T.reset_index().rename(columns={"index": "Policy"})

    # === Convert units for policy support ===
    rows_to_convert = ["Support for Electricity [GWh]", "Support for Hydrogen [GWh]"]
    mask = df_policy_all["Policy"].isin(rows_to_convert)
    df_policy_all.loc[mask, df_policy_all.columns[1:]] = df_policy_all.loc[mask, df_policy_all.columns[1:]].applymap(lambda x: round(x / 1000, 2))

    # === Convert simulation metrics (GWh, Cost) ===
    scenario_cols = df_sim_all.columns[1:]
    for i, row in df_sim_all.iterrows():
        metric = row['Metric']
        if 'GWh' in metric:
            df_sim_all.loc[i, scenario_cols] = row[scenario_cols] / 1000
        elif 'Cost' in metric:
            df_sim_all.loc[i, scenario_cols] = row[scenario_cols] / 1e6

    df_sim_all[scenario_cols] = df_sim_all[scenario_cols].round(2)

    # === Append NPV row ===
    npv_row = {"Metric": "Net Present Value (NPV) [b£]"}
    for col in scenario_cols:
        value = npv_results.get(col, None)
        if pd.notna(value):
            npv_row[col] = int(round(value))  # ⬅️ Round to nearest integer
        else:
            npv_row[col] = None
    df_sim_all = pd.concat([df_sim_all, pd.DataFrame([npv_row])], ignore_index=True)
    
    # === Save summary to Excel ===
    output_summary_path = os.path.join(main_path, "All_simulations_results.xlsx")
    with pd.ExcelWriter(output_summary_path, engine='openpyxl', mode='w') as writer:
        df_sim_all.to_excel(writer, sheet_name="Simulation Summary", index=False)
        df_policy_all.to_excel(writer, sheet_name="Policy Support Summary", index=False)

    print(f"\nAll simulation results (including NPV) saved to: {output_summary_path}")

    # === Create combined figure with 4 subplots ===
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()
    for idx, (gens_flag, scenario_name) in enumerate(plan_scenarios.items()):
        safe_name = gens_flag.replace('/', '_').replace(' ', '_')
        img_path = os.path.join(figures_folder, f"Fig_main_OPGF_gens_{safe_name}.png")
        if os.path.exists(img_path):
            img = imread(img_path)
            axs[idx].imshow(img)
            axs[idx].set_title(scenario_name, fontsize=14)
            axs[idx].axis('off')
        else:
            axs[idx].text(0.5, 0.5, f"Image not found\n{img_path}", ha='center', va='center')
            axs[idx].set_title(scenario_name, fontsize=14)
            axs[idx].axis('off')

    plt.tight_layout()
    combined_fig_path = os.path.join(figures_folder, "Fig_all_simulations_energy_mix.png")
    plt.savefig(combined_fig_path, dpi=300)
    plt.close(fig)
    print(f"Combined figure saved to: {combined_fig_path}")

    
    
    
    

def run_nash_scenario_simulation(phi_val, gens_flag, time_steps, demand_level, output_folder):
    import os
    import shutil
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    # Define scenarios and their friendly names
    plan_scenarios = {
        '25GW': "Uniform +25GW/Tech.",
        'HiRES_HiH2': "High RES–High H2",
        'HiRES_LwH2': "High RES–Low H2",
        'LwBESS_HiH2': "High H2–Low BESS"
    }

    # Empty dictionaries to collect results
    simulation_results = {}

    # Empty dict to collect NPV results
    npv_results = {}

    for gens_flag, scenario_name in plan_scenarios.items():
        print(f"\nRunning scenario: {scenario_name} ({gens_flag})")

        # Build scenario paths
        main_path = os.path.join(*output_folder.split('\\')[:5])
        last_path = os.path.join(*output_folder.split('\\')[-2:])
        name_folder = 'Case_' + gens_flag
        sim_path = os.path.join(os.path.join(main_path, name_folder, last_path))

        sim_folder = os.path.join(sim_path, "Simulation_results.xlsx")

        # === Copy and collect figures ===
        all_sim_folder_path = os.path.join(*output_folder.split('\\')[:5])
        figures_folder = os.path.join(all_sim_folder_path, "Figures")
        os.makedirs(figures_folder, exist_ok=True)

        fig_src_path = os.path.join(sim_path, "Fig_main_OPGF_gens.png")
        safe_scenario_name = gens_flag.replace('/', '_').replace(' ', '_')
        fig_dest_path = os.path.join(figures_folder, f"Fig_main_OPGF_gens_{safe_scenario_name}.png")

        if os.path.exists(fig_src_path):
            shutil.copy(fig_src_path, fig_dest_path)
            print(f"Copied figure for {scenario_name} to Figures folder.")
        else:
            print(f"Figure not found for {scenario_name}: {fig_src_path}")

        # === Extract results from Simulation_results.xlsx ===
        try:
            df_sim = pd.read_excel(sim_folder, sheet_name="Simulation_results")
        except Exception as e:
            print(f"Failed to read results for scenario {scenario_name}: {e}")
            continue

        # Collect key metrics
        simulation_results[scenario_name] = {
            "Total Demand [GWh]": df_sim.loc[df_sim["Metric"] == "Total Demand", "Value"].values[0],
            "Electric Demand [GWh]": df_sim.loc[df_sim["Metric"] == "Electric Demand", "Value"].values[0],
            "Hydrogen Demand [GWh]": df_sim.loc[df_sim["Metric"] == "Hydrogen Demand", "Value"].values[0],
            "Total Energy Supply [GWh]": df_sim.loc[df_sim["Metric"] == "Total Generation", "Value"].values[0],
            "Total Operational Cost [m£]": df_sim.loc[df_sim["Metric"] == "Total Cost", "Value"].values[0],
            "Total CO2 Emissions [Tonnes]": df_sim.loc[df_sim["Metric"] == "CO2 Emissions", "Value"].values[0],
        }

        # === Extract NPV from Generation_results_with_NPV.xlsx ===
        npv_file = os.path.join(sim_path, "Generation_results_with_NPV.xlsx")
        try:
            df_npv = pd.read_excel(npv_file, sheet_name="Total NPV Summary")
            npv_results[scenario_name] = df_npv.iloc[0]["Total System NPV (£, billion)"]
        except Exception as e:
            print(f"Failed to read NPV for scenario {scenario_name}: {e}")
            npv_results[scenario_name] = None

    # === Create pivoted DataFrame ===
    df_sim_all = pd.DataFrame(simulation_results).T
    df_sim_all.index.name = 'Scenario'
    df_sim_all = df_sim_all.T.reset_index().rename(columns={"index": "Metric"})
    scenario_cols = df_sim_all.columns[1:]

    # Apply conversion row-wise based on the Metric
    for i, row in df_sim_all.iterrows():
        metric = row['Metric']
        if 'GWh' in metric:
            df_sim_all.loc[i, scenario_cols] = row[scenario_cols] / 1000  # Convert MWh to GWh
        elif 'Cost' in metric:
            df_sim_all.loc[i, scenario_cols] = row[scenario_cols] / 1e6  # Convert £ to million £

    # Optional: round to 2 decimals
    df_sim_all[scenario_cols] = df_sim_all[scenario_cols].round(2)

    # === Append NPV row ===
    npv_row = {"Metric": "Net Present Value (NPV) [b£]"}
    for col in scenario_cols:
        value = npv_results.get(col, None)
        if pd.notna(value):
            npv_row[col] = int(round(value))  # ⬅️ Round to nearest integer
        else:
            npv_row[col] = None
    df_sim_all = pd.concat([df_sim_all, pd.DataFrame([npv_row])], ignore_index=True)
    
    # === Save to Excel ===
    output_summary_path = os.path.join(main_path, "All_simulations_results.xlsx")
    with pd.ExcelWriter(output_summary_path, engine='openpyxl', mode='w') as writer:
        df_sim_all.to_excel(writer, sheet_name="Simulation Summary", index=False)

    print(f"\nAll simulation results (including NPV) saved to: {output_summary_path}")

    # === Create combined figure with 4 subplots ===
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()

    for idx, (gens_flag, scenario_name) in enumerate(plan_scenarios.items()):
        safe_name = gens_flag.replace('/', '_').replace(' ', '_')
        img_path = os.path.join(figures_folder, f"Fig_main_OPGF_gens_{safe_name}.png")

        if os.path.exists(img_path):
            img = imread(img_path)
            axs[idx].imshow(img)
            axs[idx].set_title(scenario_name, fontsize=14)
            axs[idx].axis('off')
        else:
            axs[idx].text(0.5, 0.5, f"Image not found\n{img_path}", ha='center', va='center')
            axs[idx].set_title(scenario_name, fontsize=14)
            axs[idx].axis('off')

    plt.tight_layout()
    combined_fig_path = os.path.join(figures_folder, "Fig_all_simulations_energy_mix.png")
    plt.savefig(combined_fig_path, dpi=300)
    plt.close(fig)

    print(f"Combined figure saved to: {combined_fig_path}")






def NPV_Calculation(model, systemData, generation_file_path, sheet_name="OPGF Model"):
    import pandas as pd
    import numpy as np
    import os

    # ⚙️ Electricity price from model (fallback if not present)
    electricity_price = getattr(model, 'pE', 80)
    if hasattr(electricity_price, 'value'):
        electricity_price = electricity_price.value
    print(f"⚙️ Using electricity price: {electricity_price:.2f} £/MWh")

    # ⚙️ CO2 price from model (fallback if not present)
    co2_price = getattr(model, 'pCO2', 100)  # £/tCO2
    if hasattr(co2_price, 'value'):
        co2_price = co2_price.value
    print(f"⚙️ Using CO2 price: {co2_price:.2f} £/tCO2")

    # 1️⃣ Read generation mix
    gen_df = pd.read_excel(generation_file_path, sheet_name=sheet_name)
    gen_df.rename(columns=lambda x: x.strip(), inplace=True)

    # Normalize type names
    gen_df['Type'] = gen_df['Type'].str.strip().str.lower()
    players = systemData['Agg_Players'].copy()
    players['type'] = players['type'].str.strip().str.lower()

    # 2️⃣ Create mapping for aggregated technologies
    type_map = {
        'onshore wind': 'wind',
        'offshore wind': 'wind',
        'hydro ror': 'hydropower',
        'hydro reservoir': 'hydropower',
        'g2p (h2-ccgt)': 'g2p',
        'g2p (h2-ocgt)': 'g2p',
        'g2p (fuel cell)': 'g2p',
        'h2-storage': 'h2-storage',
        'bess': 'bess'
    }
    players['agg_type'] = players['type'].replace(type_map)
    gen_df['agg_type'] = gen_df['Type']

    merge_cols = ['id', 'agg_type', 'type', 'Capex (£/kW)', 'Fixed O&M (£/kW/year)',
                  'Marginal Cost (£/MWh)', 'economic_life', 'discount_rate', 'CF', 'emissions']

    merged = pd.merge(gen_df, players[merge_cols], on='agg_type', how='left')
    merged = merged[merged['Generation (MW)'] > 1e-3].copy()

    # 3️⃣ Fill missing entries with average of similar types
    for t in merged.loc[merged['Capex (£/kW)'].isna(), 'agg_type'].unique():
        similar = players[players['agg_type'] == t]
        if not similar.empty:
            for c in ['Capex (£/kW)', 'Fixed O&M (£/kW/year)', 'Marginal Cost (£/MWh)',
                      'economic_life', 'discount_rate', 'CF', 'emissions']:
                merged.loc[merged['agg_type'] == t, c] = similar[c].mean()
        else:
            for c in ['Capex (£/kW)', 'Fixed O&M (£/kW/year)', 'Marginal Cost (£/MWh)',
                      'economic_life', 'discount_rate', 'CF', 'emissions']:
                merged[c] = merged[c].fillna(players[c].mean())

    # Special case fix — assume CF = 0.3 for H₂-storage
    merged.loc[merged['agg_type'] == 'h2-storage', 'CF'] = 0.3

    # 4️⃣ Normalize discount rate
    def normalize_discount_rate(r):
        if pd.isna(r) or r < 0:
            return 0
        if r > 1:  # assume percentage
            return r / 100
        return r

    merged['discount_rate'] = merged['discount_rate'].apply(normalize_discount_rate)

    # 5️⃣ Capital Recovery Factor (CRF)
    if model is not None and hasattr(model, 'AF'):
        AF_dict = model.AF.extract_values()
        merged['CRF'] = merged['id'].map(AF_dict)/100
    else:
        merged['CRF'] = np.nan

    def safe_crf(x):
        if pd.notna(x['CRF']):
            return x['CRF']
        try:
            if x['economic_life'] <= 0:
                return 0
            if x['discount_rate'] > 0:
                crf = (x['discount_rate'] * (1 + x['discount_rate']) ** x['economic_life']) / \
                      ((1 + x['discount_rate']) ** x['economic_life'] - 1)
                return min(crf, 1.0)  # normalize CRF ≤ 1
            return min(1.0 / x['economic_life'], 1.0)
        except Exception:
            return 0

    merged['CRF'] = merged.apply(safe_crf, axis=1)

    # 6️⃣ Installed capacity from 1-hour peak generation
    merged['Installed Capacity (MW)'] = merged['Generation (MW)'] / merged['CF']

    # Annual generation
    merged['Annual Generation (MWh)'] = merged['Installed Capacity (MW)'] * merged['CF'] * 8760

    # Hydrogen price if exists
    hydrogen_price = getattr(model, 'pH2', electricity_price)
    if hasattr(hydrogen_price, 'value'):
        hydrogen_price = hydrogen_price.value

    # Revenue
    merged['Annual Revenue (£)'] = merged.apply(
        lambda x: x['Annual Generation (MWh)'] *
                  (hydrogen_price if any(k in x['agg_type'] for k in ['h2', 'p2g', 'g2g'])
                   else electricity_price),
        axis=1
    )

    # OPEX, CAPEX, Variable Cost
    merged['Annual OPEX (£)'] = merged['Fixed O&M (£/kW/year)'] * merged['Installed Capacity (MW)'] * 1000
    merged['Annual CAPEX (£)'] = merged['Capex (£/kW)'] * merged['Installed Capacity (MW)'] * 1000 * merged['CRF']
    merged['Variable Cost (£)'] = merged['Annual Generation (MWh)'] * merged['Marginal Cost (£/MWh)']

    # 🔹 CO2 cost
    # Extract emissions from model if available, else fallback to systemData
    if hasattr(model, 'emis'):
        emis_dict = model.emis.extract_values()
        merged['emis_factor'] = merged['id'].map(emis_dict)
        merged['emis_factor'] = merged['emis_factor'].fillna(merged['emissions'])
    else:
        merged['emis_factor'] = merged['emissions'].fillna(0)

    merged['Annual CO2 Cost (£)'] = merged['Annual Generation (MWh)'] * merged['emis_factor'] * co2_price

    # Net cashflow including CO2 cost
    merged['Net Annual Cashflow (£)'] = merged['Annual Revenue (£)'] - (
        merged['Annual OPEX (£)'] + merged['Annual CAPEX (£)'] + merged['Variable Cost (£)'] + merged['Annual CO2 Cost (£)']
    )

    # 7️⃣ NPV over economic life
    def safe_npv(x):
        if x['discount_rate'] <= 0:
            return x['Net Annual Cashflow (£)'] * x['economic_life']
        return x['Net Annual Cashflow (£)'] * (
            (1 - (1 + x['discount_rate']) ** -x['economic_life']) / x['discount_rate']
        )

    merged['NPV (£)'] = merged.apply(safe_npv, axis=1)
    merged['NPV (£, billion)'] = merged['NPV (£)'] / 1e9

    # Remove duplicate technologies
    merged = merged.drop_duplicates(subset=['Type', 'Generation (MW)'])

    # Total system NPV
    total_npv = merged['NPV (£)'].sum()
    total_row = pd.DataFrame([{
        'Type': 'System Total',
        'Generation (MW)': merged['Generation (MW)'].sum(),
        'CF': np.nan,
        'CRF': np.nan,
        'Annual Revenue (£)': merged['Annual Revenue (£)'].sum(),
        'Annual OPEX (£)': merged['Annual OPEX (£)'].sum(),
        'Annual CAPEX (£)': merged['Annual CAPEX (£)'].sum(),
        'Variable Cost (£)': merged['Variable Cost (£)'].sum(),
        'Annual CO2 Cost (£)': merged['Annual CO2 Cost (£)'].sum(),
        'Net Annual Cashflow (£)': merged['Net Annual Cashflow (£)'].sum(),
        'NPV (£, billion)': total_npv / 1e9
    }])

    npv_summary = pd.concat([merged[['Type', 'Generation (MW)', 'CF', 'CRF', 'Annual Revenue (£)',
                                      'Annual OPEX (£)', 'Annual CAPEX (£)', 'Variable Cost (£)',
                                      'Annual CO2 Cost (£)', 'Net Annual Cashflow (£)', 'NPV (£, billion)']],
                              total_row], ignore_index=True)

    # Save results
    output_path = os.path.join(os.path.dirname(generation_file_path),
                               "Generation_results_with_NPV.xlsx")
    with pd.ExcelWriter(output_path, mode='w') as writer:
        npv_summary.to_excel(writer, sheet_name="NPV Results", index=False)
        pd.DataFrame([{
            'Total System NPV (£)': total_npv,
            'Total System NPV (£, billion)': total_npv / 1e9
        }]).to_excel(writer, sheet_name="Total NPV Summary", index=False)

    print(f"✅ NPV calculation completed successfully.\nResults saved to: {output_path}")
    print(f"💰 Total System NPV = £{total_npv:,.2f}  (≈ {total_npv/1e9:.2f} billion)")
    return npv_summary
