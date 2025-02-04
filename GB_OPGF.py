# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 19:12:59 2025

@author: Mhdella
"""

# importing the libs
import pandapipes as ppipes
import pandapower as ppower
import pandapower.converter as pc
import matplotlib.pyplot as matplot
# Libraries for parallel processing
from multiprocessing import Process, Pool
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm, trange

from pandas import DataFrame
from numpy import array
from numpy.random import random
import pandas as pd
import numpy as np
import pulp as lp
import time

from geopy.distance import geodesic  # For calculating distances between coordinates


# importing the major 
from pandapipes.multinet.create_multinet import create_empty_multinet, add_net_to_multinet
from pandapipes.multinet.control.controller.multinet_control import P2GControlMultiEnergy, G2PControlMultiEnergy
from pandapipes.multinet.control.run_control_multinet import run_control

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



def data_import(Scenarios):
    # Set the path to the GB_Pw_29bus folder
    path = r"Refined FES scenario inputs/"
    GG_PW_path = "GB_Pw_29bus_Data/"  
    GB_Gas_path = "GB_Gas_NTS_Data/"  

    # Dictionary to store system data
    systemData = {        
        
        # Import the Bus, Gen, Branch, and GenCost data for the power system
        'Bus Data': pd.read_csv(GG_PW_path + 'GB_busData.csv'),
        'Gen Data': pd.read_csv(GG_PW_path + 'GB_genData.csv'),
        'Branch Data': pd.read_csv(GG_PW_path + 'GB_branchData.csv'),
        'Gen Cost': pd.read_csv(GG_PW_path + 'GB_genCost.csv'),
        'Bus Names': pd.read_csv(GG_PW_path + 'GB_busName.csv'),
        
        'Zone_Demand': pd.read_excel(GG_PW_path + 'GB_Demand.xlsx', sheet_name='Zone_Demand', index_col=0).transpose(),
        'Urban_Demand': pd.read_excel(GG_PW_path + 'GB_Demand.xlsx', sheet_name='Urban_Demand', index_col=0).transpose(),
        'Rural_Demand': pd.read_excel(GG_PW_path + 'GB_Demand.xlsx', sheet_name='Rural_Demand', index_col=0).transpose(),
        'H2_Demand': pd.read_excel(GG_PW_path + 'GB_Demand.xlsx', sheet_name='H2_Demand', index_col=0).transpose(),

        'GB Demand Profiles': pd.read_excel(GG_PW_path + 'GB_Demand.xlsx', sheet_name='Profiles', index_col=0),
        
    } 
    
    # Import gas network data
    systemData['Gas Network Data'] = {
        # Load gas nodes and demands
        'Nodes and Demands': pd.read_csv(GB_Gas_path + 'Nodes-and-demands.csv'),
        
        # Load demand profiles for gas nodes
        'Demand Profiles': {
            profile_name: pd.read_csv(f"{GB_Gas_path}/{profile_name}.csv")
            for profile_name in [
                "Demand PowerStation Profile",
                "Demand Industrial Profile",
                "Demand Interconnector Profile",
                "Demand LDZ Profile",
            ]
        },
        
        # Load pipeline details
        'Pipeline Details': pd.read_csv(GB_Gas_path + 'Pipeline-details.csv'),
    }
    
    
    # Process time series profile
    # Process P & Q (demand) for Bus data (Q is not needed, because DC power flow)
    systemData['Bus Data']['2'] = systemData['Zone_Demand'][Scenarios['Zone_Demand']].values  # P(demand)
    
    systemData['Gen Data']['8'] = systemData['Gen Data']['8']/1000  
    systemData['Gen Data']['1'] = systemData['Gen Data']['1']/1000

    
    # Normalize the values by dividing each column by its sum
    systemData['GB Demand Profiles'] = systemData['GB Demand Profiles'].div(systemData['GB Demand Profiles'].sum(axis=0), axis=1)

   
    # Add time series load profiles
    systemData['Time Series Profiles'] = {
        'e_load_time_series': systemData['GB Demand Profiles'][Scenarios['Profile']].values,
        
        'g_load_time_series': np.outer(
            systemData['H2_Demand'][Scenarios['H2_Demand']].values.astype(float),
            systemData['GB Demand Profiles']['H2 industry'].values.astype(float),
        ) 
        
    }

    
    return systemData



def power_network(time_steps, systemData):
    # Create the power system using the imported data
    ppc = {
        "version": '2',
        "baseMVA": 100.0,  # Base MVA for the system
        "bus": systemData['Bus Data'].to_numpy(),  # Bus data
        "gen": systemData['Gen Data'].to_numpy(),  # Generator data
        "branch": systemData['Branch Data'].to_numpy(),  # Branch data
        "gencost": systemData['Gen Cost'].to_numpy(),  # Generator cost data
        "bus_name": systemData['Bus Names'].to_numpy()  # Bus names
    }

    # Create the network using pandapower from the ppc (MATPOWER) format
    net_temp = pc.from_ppc(ppc, f_hz=50, validate_conversion=False)
    
    # Create an empty pandapower network with the specified parameters
    net = ppower.create_empty_network(f_hz=50, sn_mva=100)
    
    # Assign the bus data to the network
    net.bus = net_temp.bus
    bus_id = net.bus['name']

    # Loop through the branches and create lines in the network
    for i in range(ppc['branch'].shape[0]):
        ppower.create_line_from_parameters(
            net, 
            from_bus=list(bus_id).index(ppc['branch'][i][0]),
            to_bus=list(bus_id).index(ppc['branch'][i][1]),
            length_km=1, 
            r_ohm_per_km=ppc['branch'][i][2],
            x_ohm_per_km=ppc['branch'][i][3],
            c_nf_per_km=ppc['branch'][i][4], 
            max_i_ka=100
        )

    # Loop through the generators and create generators in the network
    for i in range(ppc['gen'].shape[0]):
        ppower.create_gen(
            net, 
            bus=list(bus_id).index(ppc['gen'][i][0]),
            p_mw=ppc['gen'][i][1], 
            vm_pu=1.0,
            max_q_mvar=ppc['gen'][i][3], 
            min_q_mvar=ppc['gen'][i][4],
            max_p_mw=ppc['gen'][i][8], 
            min_p_mw=ppc['gen'][i][9],
            controllable=True
        )

    # Assign the time series load profiles to the network (optional, if needed)
    net.load = net_temp.load
    load = systemData['Time Series Profiles']['e_load_time_series'][time_steps]
    net.load['p_mw'] *= load

    # Loop through the generators and apply cost data to the network
    for i in range(net.gen.shape[0]):
        ppower.create_poly_cost(
            net, 
            element=i, 
            et="gen",
            cp2_eur_per_mw2=ppc['gencost'][i][4],
            cp1_eur_per_mw=ppc['gencost'][i][5],
            cp0_eur=ppc['gencost'][i][6]
        )

    # Add external grid (slack bus)
    ppower.create_ext_grid(net, 0, min_p_mw=0, max_p_mw=0)
    
    return net



############################

# energy_gas = 39.41  # kWh/kg, energy content of hydrogen
energy_gas = 11.87  # kWh/kg, energy content of natural gas

def convert_gwh_to_mdot(gwh, energy_gas):
    # Convert GWh to mass flow rate in kg/s based on energy content (assume energy_gas is in kWh/kg)
    return (gwh * 1e3) / (energy_gas * 3600)


def net_gas(time_steps, systemData):
    # Extract gas network data from systemData
    gas_data = systemData['Gas Network Data']
    nodes_data = gas_data['Nodes and Demands']
    demand_profiles_data = gas_data['Demand Profiles']
    pipelines_data = gas_data['Pipeline Details']
    
    # Create the empty gas network with the correct fluid type
    net = ppipes.create_empty_network(fluid="lgas")  

    temp_system = 288.15  # System temperature in Kelvin
    energy_gas = 11.87  # kWh/kg, energy content of natural gas

    # Helper function to convert GWh to mass flow rate
    def convert_gwh_to_mdot(gwh, energy_gas):
        return (gwh * 1e3) / (energy_gas * 3600)

    # Dictionary to store junctions in the network
    junctions = {}

    # Create Junctions for Each Node
    for index, row in nodes_data.iterrows():
        node_name = row['NAME']
        node_flow = row['NodeFlowByCategory(Base Thermal)']
        profile_name = row['NodeFlowProfileNameByCategory(Base Thermal)']
        profile_data = demand_profiles_data.get(profile_name)

        # Ensure junction is created for every node
        if node_name not in junctions:
            junctions[node_name] = ppipes.create_junction(
                net, pn_bar=1, tfluid_k=temp_system, name=node_name
            )
        
        if profile_data is not None:
            # Extract the Y column from the profile data
            y_values = profile_data['Y'].values  # assuming Y contains demand/supply proportions for each hour
            
            # Calculate the total demand or supply for the node based on the profile data
            total_demand_or_supply = node_flow * sum(y_values)  # Sum the proportions for total demand/supply
            
            # Create a single sink or source based on total demand/supply
            if node_flow < 0:  # Negative flow means demand, treated as a sink
                mdot_kg_per_s = convert_gwh_to_mdot(abs(total_demand_or_supply), energy_gas)
                ppipes.create_sink(
                    net, junction=junctions[node_name],
                    mdot_kg_per_s=mdot_kg_per_s,
                    name=f"Sink {node_name}_Total"
                )
            elif node_flow > 0:  # Positive flow means supply, treated as a source
                mdot_kg_per_s = convert_gwh_to_mdot(total_demand_or_supply, energy_gas)
                ppipes.create_source(
                    net, junction=junctions[node_name],
                    mdot_kg_per_s=mdot_kg_per_s,
                    name=f"Source {node_name}_Total"
                )
        else:
            # If no demand profile is found, directly use NodeFlowByCategory(Base Thermal) value
            total_demand_or_supply = node_flow
            if node_flow < 0:  # Negative flow means demand, treated as a sink
                mdot_kg_per_s = convert_gwh_to_mdot(abs(total_demand_or_supply), energy_gas)
                ppipes.create_sink(
                    net, junction=junctions[node_name],
                    mdot_kg_per_s=mdot_kg_per_s,
                    name=f"Sink {node_name}_NoProfile"
                )
            elif node_flow > 0:  # Positive flow means supply, treated as a source
                mdot_kg_per_s = convert_gwh_to_mdot(total_demand_or_supply, energy_gas)
                ppipes.create_source(
                    net, junction=junctions[node_name],
                    mdot_kg_per_s=mdot_kg_per_s,
                    name=f"Source {node_name}_NoProfile"
                )

    # Create pipes based on pipeline details
    for _, row in pipelines_data.iterrows():
        from_node = row["FacilityFromNodeName"]
        to_node = row["FacilityToNodeName"]
        
        # Ensure that both nodes exist in junctions before creating the pipe
        if from_node not in junctions:
            print(f"Warning: {from_node} not found in junctions!")
            continue
        if to_node not in junctions:
            print(f"Warning: {to_node} not found in junctions!")
            continue
        
        pipe_length_km = row["PipeLength"]  # Use PipeLength from the pipeline details
        ppipes.create_pipe_from_parameters(
            net, from_junction=junctions[from_node], to_junction=junctions[to_node],
            length_km=pipe_length_km, k_mm=0.025, diameter_m=row["PipeDiameter"] / 1000,
            name=f"{from_node}-{to_node}"
        )
    
    # Add an external grid if none exists
    if len(net.ext_grid) == 0:
        first_junction = list(junctions.values())[0]
        ppipes.create_ext_grid(
            net, junction=first_junction, p_bar=75, t_k=temp_system, name="Default Grid Connection"
        )
    
    return net


#########

def plot(net_gas, net_power):
    # plot network
    ppipes.plotting.simple_plot(net_gas, plot_sinks=True, plot_sources=True)
    global xy
    xy = ppipes.plotting.pressure_profile_to_junction_geodata(net_gas)
    matplot.plot(xy['y'])
    matplot.ylim(70,80)
    
    
    ppower.plotting.simple_plot(net_power)
    ppower.plotting.plotly.pf_res_plotly(net_power)
    ppower.plotting.plotly.vlevel_plotly(net_power)
    

############################


def run_OPF():
    # Define the time steps and generate the power network
    # time_steps = 0  # For simplicity, using the first time step
    net_power = power_network(time_steps)

    # Solve the DC OPF
    ppower.rundcopp(net_power)

    # Display results
    print("\nDC OPF Results:")
    print(net_power.res_gen)
    print(net_power.res_bus)
    print(net_power.res_line)

    # Plot network
    ppower.plotting.simple_plot(net_power)

    return net_power


def run_OPGF(t):
        
    
    # Run the simulation
    run_control(multinet, ctrl_variables= {'nets': {'power':{'run': ppower.rundcopp}}}) # ctrl_variables added to run the dc opf
    
    # run_control(multinet)
    

    return multinet



def form_multinet(net_power, net_gas, systemData):
    # create multinet and add networks:
    multinet = create_empty_multinet('GB_multinet')
    add_net_to_multinet(multinet, net_power, 'power')
    add_net_to_multinet(multinet, net_gas, 'gas')
    
    
    return multinet





def choose_scenario_from_list(options_dict, choice):
    """
    Given a dictionary of options and a choice (number), return the corresponding scenario option.
    """
    if choice in options_dict:
        return options_dict[choice]
    else:
        raise ValueError(f"Invalid choice: {choice}. Please select a number between {min(options_dict)} and {max(options_dict)}.")


if __name__ == "__main__":
    # Scenario options dictionaries
    H2_options = {
        1: 'GB H2 industrial demand',
        2: 'GB H2 demand from transport and shipping',
        3: 'Total'
    }
    
    Zone_options = {
        0: '29Bus',
        1: 'GB domestic demand distribution', 2: 'GB non-domestic demand distribution',
        3: 'GB domestic space heating', 4: 'GB non-domestic space heating',
        5: 'GB domestic water heating', 6: 'GB non-domestic water heating',
        7: 'GB domestic electric load (non transport/non heat)',
        8: 'GB non-domestic electric load (non transport/non heat)',
        9: 'GB domestic electric transport', 10: 'GB non-domestic electric transport',
        11: 'GB electric rail', 12: 'GB domestic appliances',
        13: 'GB industrial demand distribution', 14: 'GB industrial electric demand',
        15: 'GB H2 industrial demand ', 16: 'GB H2 demand from transport and shipping',
        17: 'GB cooking', 18: 'GB cooling', 19: 'GB agriculture H2 demand',
        20: 'GB agriculture electricity demand', 21: 'Total'
    }

    Profile_options = {
        1: 'DSH', 2: 'DWH', 3: 'CSH', 4: 'CWH',
        5: 'Other', 6: 'EV', 7: 'SA', 8: 'Rail', 9: 'H2 industry'
    }

    # scenario_options = [H2 option, Zone option, and Profile option]
    scenario_options = [3, 0, 9]  
    
    # scenario_options = [1, 21, 5]  
    # scenario_options = [2, 21, 5] 
    # scenario_options = [3, 21, 5] 
    # scenario_options = [3, 21, 9]  


    # Extract the corresponding scenario values
    try:
        H2_Demand = choose_scenario_from_list(H2_options, scenario_options[0])
        Zone_Demand = choose_scenario_from_list(Zone_options, scenario_options[1])
        Profile = choose_scenario_from_list(Profile_options, scenario_options[2])
    except ValueError as e:
        print(e)
        exit(1)

    # Construct the Scenarios dictionary
    Scenarios = {
        'H2_Demand': H2_Demand,
        'Zone_Demand': Zone_Demand,
        'Profile': Profile,
    }

    # print("\nSelected Scenarios:")
    # for key, value in Scenarios.items():
    #     print(f"{key}: {value}")

#################################################

    time_steps = 0
    # time_steps = 1
    # time_steps = 2
    # time_steps = 12
    
    
    systemData = data_import(Scenarios)
    
    net_power = power_network(time_steps, systemData)
    net_gas = net_gas(time_steps, systemData)

    multinet = form_multinet(net_power, net_gas, systemData)
    
    run_OPGF(time_steps)

    
    
    print()
    print('res_gen=', multinet['nets']['power']['res_gen']['p_mw'])
    print()
    print('res_bus_va_degree=', multinet['nets']['power']['res_bus']['va_degree'])
    print()
    print('res_line_loading_percent', multinet['nets']['power']['res_line']['loading_percent'])
    print()
    print('res_load=', multinet['nets']['power']['res_load']['p_mw'])
    print()    
    print('res_sink=', multinet.nets['gas'].res_sink['mdot_kg_per_s'])
    print()
    # print('res_source=', multinet.nets['gas'].res_source['mdot_kg_per_s'])
    print()
    print('res_p_bar=', multinet.nets['gas']. res_junction['p_bar'])
    print()    
    print('res_pipes', multinet.nets['gas'].res_pipe['vdot_norm_m3_per_s'])
    print()
    print('res_ext_grid', multinet.nets['gas'].res_ext_grid['mdot_kg_per_s'])
    print()


    results = {
        # "Electrolyser out": multinet.nets['hydrogen'].source['mdot_kg_per_s'][0],
        "Maximum generation capacity": sum(multinet.nets['power'].gen['max_p_mw']),
        "Resultant generation": sum(multinet.nets['power'].res_gen['p_mw']),
        "Total and resultant load": sum(multinet.nets['power'].res_load['p_mw']),
        "External grid": multinet.nets['power'].res_ext_grid['p_mw'][0],
        "Losses": sum(multinet.nets['power'].res_line['pl_mw']),
        # "Supply - Demand": difference,
        
        # "Gas demand (kg/s)": sum(multinet.nets['gas'].res_sink['mdot_kg_per_s']),
        # "Gas supply (kg/s)": sum(multinet.nets['gas'].res_source['mdot_kg_per_s']),
        # "Gas external (kg/s)": sum(multinet.nets['gas'].res_ext_grid['mdot_kg_per_s']),
        
        "Gas demand (GWh)": sum(multinet.nets['gas'].res_sink['mdot_kg_per_s'])*(energy_gas*3600)/1e3,
        "Gas supply (GWh)": sum(multinet.nets['gas'].res_source['mdot_kg_per_s'])*(energy_gas*3600)/1e3,
        "Gas external (GWh)": sum(multinet.nets['gas'].res_ext_grid['mdot_kg_per_s'])*(energy_gas*3600)/1e3,

    }
        
    
    from IPython.display import display
    display(results)
    
    print("\nSelected Scenarios:")
    for key, value in Scenarios.items():
        print(f"{key}: {value}")

    
    # plot(net_gas, net_power)
    
    # ppipes.pipeflow(net_gas)
    # print(net_gas.res_junction)
    # print(net_gas.res_pipe['vdot_norm_m3_per_s'])

