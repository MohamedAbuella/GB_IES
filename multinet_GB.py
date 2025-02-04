# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 06:05:41 2025

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

#start_time = time.time()

# importing the major 
from pandapipes.multinet.create_multinet import create_empty_multinet, add_net_to_multinet
from pandapipes.multinet.control.controller.multinet_control import P2GControlMultiEnergy, G2PControlMultiEnergy, GasToGasConversion
from pandapipes.multinet.control.run_control_multinet import run_control

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def data_import(Scenarios):
    path = r"Refined FES scenario inputs/"
    GBdata_path = "GB_2050_Data/"  # Path to the Data folder

    systemData = {
        # 'Carbon Price': pd.read_excel(path + 'Carbon price/carbon_price.xlsx'),
        # 'Installed Capacity': pd.read_excel(path + 'InstalledCapacity/installed_capacity.xlsx', index_col=0),
        # 'Power Demand': pd.read_excel(path + 'PowerDemand_Heating_Input/powerDemand.xlsx', index_col=0),
        
        'Players': pd.read_csv(GBdata_path + 'GB_market_players.csv'), #import the parameters

        
        'Zone_Demand': pd.read_excel(GBdata_path + 'GB_Demand.xlsx', sheet_name='Zone_Demand', index_col=0).transpose(),
        'Urban_Demand': pd.read_excel(GBdata_path + 'GB_Demand.xlsx', sheet_name='Urban_Demand', index_col=0).transpose(),
        'Rural_Demand': pd.read_excel(GBdata_path + 'GB_Demand.xlsx', sheet_name='Rural_Demand', index_col=0).transpose(),
        'H2_Demand': pd.read_excel(GBdata_path + 'GB_Demand.xlsx', sheet_name='H2_Demand', index_col=0).transpose(),

        'GB Demand Profiles': pd.read_excel(GBdata_path + 'GB_Demand.xlsx', sheet_name='Profiles', index_col=0),
        'Bus Data': pd.read_csv(GBdata_path + 'GB_busData.csv'),
        'Gen Data': pd.read_csv(GBdata_path + 'GB_genData.csv'),
        'Branch Data': pd.read_csv(GBdata_path + 'GB_branchData.csv'),
        'Gen Cost': pd.read_csv(GBdata_path + 'GB_genCost.csv'),
        'Bus Names': pd.read_csv(GBdata_path + 'GB_busName.csv'),
        'P2G Data': pd.read_excel(GBdata_path + 'GB_P2G.xlsx', sheet_name=None),
        'CCS Data': pd.read_excel(GBdata_path + 'GB_CCS.xlsx', sheet_name=None),  
        
    }
    
    # Process P & Q (demand) for Bus data (Q is not needed, because DC power flow)
    systemData['Bus Data']['2']=systemData['Zone_Demand'][Scenarios['Zone_Demand']].values # P(demand)
    # systemData['Bus Data']['3']= 0.3 * systemData['Zone_Demand'][Scenarios['Zone_Demand']].values  # Q(demand)
        
    systemData['Gen Data']['8'] = systemData['Gen Data']['8']

    # systemData['Gen Data']['8'] = systemData['Gen Data']['8']/1000
    # systemData['Gen Data']['1'] = systemData['Gen Data']['1']/1000
    
    # Normalize the values by dividing each column by its sum
    systemData['GB Demand Profiles'] = systemData['GB Demand Profiles'].div(systemData['GB Demand Profiles'].sum(axis=0), axis=1)*1000


    # # Process demand data  from TWh to GWh
    # for key in ['Zone_Demand', 'Urban_Demand', 'Rural_Demand', 'H2_Demand']:
    #     demand_df = systemData[key]
    #     numeric_columns = demand_df.columns.difference(['Category'], sort=False)
    #     demand_df[numeric_columns] = demand_df[numeric_columns].apply(pd.to_numeric, errors='coerce') * 1000
        
    #     systemData[key] = demand_df
    
    
    # Process time series profile
    systemData['Time Series Profiles'] = {
        'e_load_time_series': systemData['GB Demand Profiles'][Scenarios['Profile']].values,
        
        'g_load_time_series': systemData['GB Demand Profiles']['H2 industry'].values,
                                         }
    
    systemData['Scenarios_id'] =Scenarios

    return systemData



# Coordinates for the regions (17 regions) with acronyms
regions = {
    "EE": (52.3000, 0.7000),  
    "EM": (52.9500, -0.9500),  
    "L": (51.5099, -0.1181), 
    "NW": (53.3000, -3.0000), 
    "WM": (52.5000, -1.9167),  
    "NEE": (54.9000, -1.5000), 
    "NWE": (53.7500, -2.5000),  
    "NS": (57.5000, -4.0000),  
    "SE": (51.0000, -1.0000),  
    "SW": (51.8333, -3.2500), 
    "SWE": (50.8000, -3.5000), 
    "SEE": (51.2000, 0.7000),  
    "Y": (53.9000, -1.4000),  
    "SS": (55.8667, -3.5000),  
    "CE": (50.5000, 8.0000),  
    "IE": (53.5000, -7.0000), 
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




def power_network(time_steps, systemData):
    ppc = {
        "version": '2',
        "baseMVA": 100.0,
        "bus": systemData['Bus Data'].to_numpy(),
        "gen": systemData['Gen Data'].to_numpy(),
        "branch": systemData['Branch Data'].to_numpy(),
        "gencost": systemData['Gen Cost'].to_numpy(),
        "bus_name": systemData['Bus Names'].to_numpy()
    }

    net_temp = pc.from_ppc(ppc, f_hz=50, validate_conversion=False)
    net = ppower.create_empty_network(f_hz=50, sn_mva=100)
    net.bus = net_temp.bus
    bus_id = net.bus['name']

    for i in range(ppc['branch'].shape[0]):
        ppower.create_line_from_parameters(net, from_bus=list(bus_id).index(ppc['branch'][i][0]),
                                           to_bus=list(bus_id).index(ppc['branch'][i][1]),
                                           length_km=1, r_ohm_per_km=ppc['branch'][i][2],
                                           x_ohm_per_km=ppc['branch'][i][3],
                                           c_nf_per_km=ppc['branch'][i][4], max_i_ka=100)

    for i in range(ppc['gen'].shape[0]):
        ppower.create_gen(net, bus=list(bus_id).index(ppc['gen'][i][0]),
                          p_mw=ppc['gen'][i][1], vm_pu=1.0,
                          max_q_mvar=ppc['gen'][i][3], min_q_mvar=ppc['gen'][i][4],
                          max_p_mw=ppc['gen'][i][8], min_p_mw=ppc['gen'][i][9],
                          controllable=True)

    net.load = net_temp.load
    load = systemData['Time Series Profiles']['e_load_time_series'][time_steps]
    net.load['p_mw'] *= load

    for i in range(net.gen.shape[0]):
        ppower.create_poly_cost(net, element=i, et="gen",
                                cp2_eur_per_mw2=ppc['gencost'][i][4],
                                cp1_eur_per_mw=ppc['gencost'][i][5],
                                cp0_eur=ppc['gencost'][i][6])

    ppower.create_ext_grid(net, 0, min_p_mw=0, max_p_mw=0)
    return net




# Constants
energy_hydrogen = 39.41  # kWh/kg, energy content of hydrogen
energy_gas = 14.64 # kWh/kg, energy content of natural gas

# Function to calculate distance between two coordinates
def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).kilometers

# Convert TWh to mdot
def convert_mwh_to_mdot(mwh, energy_hydrogen):
    """Convert TWh to mdot_kg_per_s using energy content of hydrogen."""
    return (mwh * 1e3) / (energy_hydrogen * 3600)




def gas_network(time_steps, systemData):
    
    net = ppipes.create_empty_network(fluid="lgas")

    temp_system = 288.15  # System temperature in Kelvin
    
    # Create junctions based on regions
    junctions = {}
    for acronym, coords in regions.items():
        junctions[acronym] = ppipes.create_junction(
            net, pn_bar=1, tfluid_k=temp_system, name=acronym, geodata=coords
        )

    # Create pipelines for gas transport
    for from_acronym, to_acronym, capacity in connections:
        from_coords = regions[from_acronym]
        to_coords = regions[to_acronym]
        distance_km = calculate_distance(from_coords, to_coords)
        ppipes.create_pipe_from_parameters(
            net, from_junction=junctions[from_acronym], to_junction=junctions[to_acronym],
            length_km=distance_km, k_mm=0.025, diameter_m=1.0, name=f"{from_acronym}-{to_acronym}"
        )

    # Process ATRCCS and BECCS Data
    atrccs_data = systemData['CCS Data']['ATRCCS']
    beccs_data = systemData['CCS Data']['BECCS']
    
    # Convert capacity from GW to MW
    atrccs_data['Capacity(MW)'] = atrccs_data['Capacity(GW)'] * 1000
    beccs_data['Capacity(MW)'] = beccs_data['Capacity(GW)'] * 1000
    
    # Compute gas input in MW
    atrccs_data['Input_MW'] = atrccs_data['Capacity(MW)'] / atrccs_data['Efficiency']
    beccs_data['Input_MW'] = beccs_data['Capacity(MW)'] / beccs_data['Efficiency']

    # Function to convert MW to mass flow rate (kg/s)

    # Assign each ATRCCS and BECCS unit to its respective junction as a sink
    for _, row in atrccs_data.iterrows():
        region = row['Region']
        input_mdot = convert_mwh_to_mdot(row['Input_MW'], energy_hydrogen)  # Convert MW -> TW before using function
        ppipes.create_sink(
            net, junction=junctions[region],
            mdot_kg_per_s=input_mdot, name=f"Sink {region}_ATRCCS"
        )

    for _, row in beccs_data.iterrows():
        region = row['Region']
        input_mdot = convert_mwh_to_mdot(row['Input_MW'], energy_hydrogen)  # Convert MW -> TW before using function
        ppipes.create_sink(
            net, junction=junctions[region],
            mdot_kg_per_s=input_mdot, name=f"Sink {region}_BECCS"
        )
        
    ppipes.create_ext_grid(net, junction=junctions["NS"], p_bar=75, t_k=temp_system, name="Grid Connection SS")


    return net






# Constants
energy_hydrogen = 39.41  # kWh/kg, energy content of hydrogen
# energy_gas = 14.64 # kWh/kg, energy content of natural gas


# Function to calculate distance between two coordinates
def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).kilometers


# Convert TWh to mdot
def convert_twh_to_mdot(twh, energy_hydrogen):
    """Convert TWh to mdot_kg_per_s using energy content of hydrogen."""
    return (twh * 1e6) / (energy_hydrogen * 3600)




def hydrogen_network(time_steps, systemData):
    
    net = ppipes.create_empty_network(fluid="hydrogen")
    # net = ppipes.create_empty_network(fluid="lgas")
    
    temp_system = 288.15  # System temperature in Kelvin
    
    Scenarios = systemData['Scenarios_id']
    
    demand_g = systemData['H2_Demand'][Scenarios['H2_Demand']]
    load_g = systemData['Time Series Profiles']['g_load_time_series'][time_steps]
    demand_g  *= load_g
    demand_data =list(demand_g)

    junctions = {}
    for acronym, coords in regions.items():
        junctions[acronym] = ppipes.create_junction(
            net, pn_bar=1, tfluid_k=temp_system, name=acronym, geodata=coords
        )

    for from_acronym, to_acronym, capacity in connections:
        from_coords = regions[from_acronym]
        to_coords = regions[to_acronym]
        distance_km = calculate_distance(from_coords, to_coords)
        ppipes.create_pipe_from_parameters(
            net, from_junction=junctions[from_acronym], to_junction=junctions[to_acronym],
            length_km=distance_km, k_mm=0.025, diameter_m=1.0, name=f"{from_acronym}-{to_acronym}"
        )

    for i, (region, demand_twh) in enumerate(zip(regions.keys(), demand_data)):
        mdot_kg_per_s = convert_twh_to_mdot(demand_twh, energy_hydrogen)
        ppipes.create_sink(
            net, junction=junctions[region],
            # mdot_kg_per_s=systemData['Time Series Profiles']['g_load_time_series'][i][time_steps] * mdot_kg_per_s,
            mdot_kg_per_s = mdot_kg_per_s,

            name=f"Sink {region}"
        )

    ppipes.create_ext_grid(net, junction=junctions["SS"], p_bar=75, t_k=temp_system, name="Grid Connection SS")
    

    return net



def coupling_p2g(net_power, net_hyd, multinet, systemData):
    # Load the P2G data from systemData
    p2g_data = systemData['P2G Data']

    # Create a mapping from regions to bus numbers in net_power
    region_to_bus = {region: int(bus) for region, bus in zip(regions.keys(), net_power['bus']['name'] - 1)}

    # Iterate over each electrolyser type (SOE, Alkaline, PEM)
    for sheet_name, df in p2g_data.items():
        print(f"Processing P2G data for {sheet_name}...")

        for _, row in df.iterrows():
            region = row["Region"]
            capacity_gw = row["Capacity(GW)"]  # Installed capacity (GW)
            efficiency = float(row["Efficiency"])  # Efficiency factor

            if region in regions:
                # Find corresponding power bus
                power_bus = region_to_bus[region]

                # Create the load in the power network (with dynamic p_mw control)
                # The load will be set to the installed capacity (capacity_gw * 1000)
                # This represents the maximum potential power consumption for this P2G unit
                p2g_power_load = ppower.create_load(
                    net_power, 
                    bus=power_bus, 
                    p_mw=capacity_gw * 1000,  # Set initial power flow (MW), controller will adjust based on efficiency
                    name=f"P2G_{region}_{sheet_name} consumption"
                )

                # Create the hydrogen source in the gas network
                hydrogen_junction = net_hyd['junction'].loc[net_hyd['junction']['name'] == region].index[0]
                p2g_gas_source = ppipes.create_source(
                    net_hyd, 
                    junction=hydrogen_junction, 
                    mdot_kg_per_s=0,  # Initially 0, controller will adjust the hydrogen flow based on power consumption
                    name=f"P2G_{region}_{sheet_name} feed in"
                )
                
                
                # Create the P2G control with proper efficiency usage
                P2GControlMultiEnergy(
                    multinet, 
                    element_index_power=p2g_power_load, 
                    element_index_gas=p2g_gas_source,
                    efficiency=efficiency,  # Apply the efficiency factor for power-to-hydrogen conversion
                    name_power_net="power", 
                    name_gas_net="hydrogen"
                )
            else:
                print(f"Warning: Region {region} not found in the bus/junction mapping. Skipping...")




def coupling_ccs(net_power, net_gas, net_hyd, multinet, systemData):
    
    # Load CCS data
    ccs_data = systemData['CCS Data']
    
    # Map each region to its corresponding power bus
    region_to_bus = {region: int(bus) for region, bus in zip(regions.keys(), net_power['bus']['name'] - 1)}
    
    # Map each region to its corresponding junction in the gas network
    region_to_gas_junction = {row['name']: idx for idx, row in net_gas['junction'].iterrows()}
    
    for sheet_name, df in ccs_data.items():
        print(f"Processing CCS data for {sheet_name}...")

        for _, row in df.iterrows():
            region = row["Region"]
            capacity_gw = row["Capacity(GW)"]
            efficiency = float(row["Efficiency"])

            if region in regions:
                power_bus = region_to_bus[region]
                
                # Create a power load in the electricity network
                ccs_power_load = ppower.create_load(
                    net_power, 
                    bus=power_bus, 
                    p_mw=capacity_gw, 
                    name=f"CCS_{region}_{sheet_name} consumption"
                )

                # Locate junctions in both gas and hydrogen networks
                hydrogen_junction = net_hyd['junction'].loc[net_hyd['junction']['name'] == region].index[0]
                gas_junction = region_to_gas_junction.get(region)  # Get the correct gas junction

                if gas_junction is None:
                    print(f"Warning: No gas junction found for region {region}. Skipping...")
                    continue  # Skip if no corresponding gas junction is found

                # Create a natural gas sink (gas is consumed)
                ccs_gas_sink = ppipes.create_sink(
                    net_gas, 
                    junction=gas_junction, 
                    mdot_kg_per_s=0,  # Controlled dynamically
                    name=f"CCS_{region}_{sheet_name} NG consumption"
                )

                # Create a hydrogen source (produced gas)
                ccs_gas_source = ppipes.create_source(
                    net_hyd, 
                    junction=hydrogen_junction, 
                    mdot_kg_per_s=0,  # Controlled dynamically
                    name=f"CCS_{region}_{sheet_name} H2 feed-in"
                )

                # Add Gas-to-Gas conversion controller
                GasToGasConversion(
                    multinet=multinet, 
                    element_index_from=ccs_gas_sink,  # Natural gas input
                    element_index_to=ccs_gas_source,  # Hydrogen output
                    efficiency=efficiency,  
                    name_gas_net_from="gas",  
                    name_gas_net_to="hydrogen"
                )

            else:
                print(f"Warning: Region {region} not found in mapping. Skipping...")

    multinet.nets['gas'].sink['mdot_kg_per_s'][24:48] = multinet.nets['gas'].sink['mdot_kg_per_s'][:24]
    multinet.nets['gas'].sink['mdot_kg_per_s'][:24] = 0



def OPGF(multinet,net_power, net_hyd):
    
    ele_opex_price = 33.69 # £/MWh
    gas_opex_price = 9.73 # £/MWh
    hyd_opex_price = 9 # £/MWh
    
    # ele_opex_price = 250 # £/MWh
    # gas_opex_price = 250 # £/MWh
    # hyd_opex_price = 250 # £/MWh

    
    # genData = pd.read_csv('Data/genData.csv')
    genData = systemData['Gen Data']['8'] 
    genCount = np.shape(genData)[0]
        
    prob = lp.LpProblem("OPGF", lp.LpMinimize)
    
    # Variables
    q_e   = lp.LpVariable.dicts("elect", list(np.arange(genCount)), cat='Continuous')
    q_g   = lp.LpVariable("gas", cat='Continuous')
    q_p2g = lp.LpVariable.dicts("P2G",list(np.arange(51)), cat='Continuous')
    # q_g2p = lp.LpVariable.dicts("G2P", list(np.arange(70)), cat='Continous')
    q_g2g = lp.LpVariable.dicts("G2G", list(np.arange(24)), cat='Continous')
    q_h   = lp.LpVariable("hydrogen", cat='Continuous')
        
    # Objective function
    prob += (
          lp.lpSum(ele_opex_price * q_e[gen] for gen in range(genCount))
        + lp.lpSum(ele_opex_price * q_p2g[p2g] for p2g in range(51))
        # + lp.lpSum(gas_opex_price * q_g2p[g2p] for g2p in range(70))
        + lp.lpSum(gas_opex_price * q_g2g[g2g] for g2g in range(24))
        + lp.lpSum(gas_opex_price * q_g)
        + lp.lpSum(hyd_opex_price * q_h)
        ), "objective"
    
    for i in range(genCount):
        prob += q_e[i] <= genData['8'][i]
        prob += q_e[i] >= 0
    
    prob += q_g <= 1e20 # limit for the suppliers
        
    p2g_units = multinet.nets['power'].load.dropna(subset=['name'])  # Remove NaN values in 'name' column
    p2g_units = p2g_units[p2g_units['name'].str.contains("P2G", na=False)]  # Apply the filter for P2G units

    p2g_limits = p2g_units['p_mw'].values  # Get the power limits (p_mw values)
    
    for i in range(len(p2g_limits)):  # Iterate over all P2G units
        prob += q_p2g[i] <= p2g_limits[i]  # Use the dynamic limits
        prob += q_p2g[i] >= 0
        
    
    g2g_units = multinet.nets['hydrogen'].source.dropna(subset=['name'])  # Remove NaN values in 'name' column
    g2g_units = g2g_units[g2g_units['name'].str.contains("CCS", na=False)]  # Filter for 'CCS' units
    
    # # Extract the flow limits for each G2G unit based on the filtered data
    # g2g_limits = []
    # for i in range(len(g2g_units)):
    #     # # Extract the mdot_kg_per_s for each G2G unit, which is the flow limit
    #     # limit = g2g_units.iloc[i]['mdot_kg_per_s'] * (energy_hydrogen*3600)/1e6  # Apply the conversion factor
    #     limit = g2g_units.iloc[i]['mdot_kg_per_s']
    #     g2g_limits.append(limit)
    
    g2g_limits = list(multinet.nets['gas'].sink['mdot_kg_per_s'][g2g_units.index])
        
    # Apply the constraints for each G2G unit in the optimization problem
    for i in range(len(g2g_limits)):
        prob += q_g2g[i] <= g2g_limits[i]  # Set upper limit for each G2G unit
        prob += q_g2g[i] >= 0  # Ensure the flow is non-negative
        
    
        
    prob += q_h <= 1e20
        
    
    prob += lp.lpSum(q_e[gen] for gen in range(genCount))  == (sum(multinet.nets['power'].load['p_mw'][0:32]) 
                                                         + lp.lpSum(q_p2g[p2g] for p2g in range(4)))
    prob += q_g  == (sum(multinet.nets['gas'].sink['mdot_kg_per_s'][0:15])*(energy_hydrogen*3600)/(1000)
                                                         + lp.lpSum(q_g2p[g2p] for g2p in range(5)))
    prob += q_h  == 2, "C3" 
    
    #prob += lp.lpSum(q_p2g[p2g] for p2g in range(4)) <= sum(p2g_limits), "C4"
    #prob += lp.lpSum(q_g2p[g2p] for g2p in range(5)) <= sum(g2p_limits), "C5"
    
    #Solve the problem using the default solver
    prob.solve(lp.PULP_CBC_CMD(msg=0))
    
    
    
    global variable
    variable = prob.variables()
    if variable[35].varValue > 0:
        electrolyser(net_power, net_hyd, multinet, variable[34].varValue)
        fuel_cell(net_power, net_hyd, multinet, 0)
    else:
        electrolyser(net_power, net_hyd, multinet, 0)
        fuel_cell(net_power, net_hyd, multinet, variable[34].varValue)
        
    # Print the value of the objective
    # print("objective=", lp.value(prob.objective))
    
    

def declaration():
    global e_generation, g_supply, hydrogen_conv
    e_generation = []
    g_supply = []
    hydrogen_conv = []
    
    
    
def run_OPGF(t, genData,  systemData):
    
    # Step 1: Form the networks
    net_power   = power_network(t, systemData)
    net_gas     = gas_network(t, systemData)
    net_hyd     = hydrogen_network(t, systemData)

    
    # Step 2: Form the multinet
    global multinet
    multinet = form_multinet(net_power, net_gas, net_hyd, systemData)
    
    # # Step 3: Initial calculations for the OPDG
    # OPGF(multinet,net_power, net_hyd)
        
    
    # Run the simulation
    run_control(multinet, ctrl_variables= {'nets': {'power':{'run': ppower.rundcopp}}}) # ctrl_variables added to run the dc opf
    
    # run_control(multinet)
    

    return multinet

def initial_run_OPGF(systemData):
    # Running the GT model
 
    # genData = pd.read_csv('Data/genData.csv')
    genData = systemData['Gen Data']['8'] 
    
    hour_of_day = 24*12*7 # Input how many hours the model shall run
    multinet = run_OPGF(hour_of_day, genData,  systemData) # Running the OPGF
    
    # global time_steps
    # time_steps = hour_of_day-1
    
    return multinet



def form_multinet(net_power, net_gas, net_hyd, systemData):
    # create multinet and add networks:
    multinet = create_empty_multinet('GB_multinet')
    add_net_to_multinet(multinet, net_power, 'power')
    add_net_to_multinet(multinet, net_gas, 'gas')
    add_net_to_multinet(multinet, net_hyd, 'hydrogen')
    
    # ## Add coupling components to the network
    # coupling_p2g(net_power, net_hyd, multinet, systemData)
    coupling_ccs(net_power, net_gas, net_hyd, multinet, systemData)

    
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
    # scenario_options = [1, 21, 5]  
    # scenario_options = [2, 21, 5] 
    # scenario_options = [3, 21, 5] 
    scenario_options = [3, 21, 9]  



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
    multinet = initial_run_OPGF(systemData)
    
    
    print()
    print('res_gen=', multinet['nets']['power']['res_gen']['p_mw'])
    print()
    print('res_bus_va_degree=', multinet['nets']['power']['res_bus']['va_degree'])
    print()
    print('res_line_loading_percent', multinet['nets']['power']['res_line']['loading_percent'])
    print()
    print('res_load=', multinet['nets']['power']['res_load']['p_mw'])
    print()    
    print('res_sink=', multinet.nets['hydrogen'].res_sink['mdot_kg_per_s'])
    print()
    # print('res_source=', multinet.nets['hydrogen'].res_source['mdot_kg_per_s'])
    print()
    print('res_p_bar=', multinet.nets['hydrogen']. res_junction['p_bar'])
    print()    
    print('res_pipes', multinet.nets['hydrogen'].res_pipe['vdot_norm_m3_per_s'])
    print()
    print('res_ext_grid', multinet.nets['hydrogen'].res_ext_grid['mdot_kg_per_s'])
    print()


    results = {
        # "Electrolyser out": multinet.nets['hydrogen'].source['mdot_kg_per_s'][0],
        "Maximum generation capacity": sum(multinet.nets['power'].gen['max_p_mw']),
        "Resultant generation": sum(multinet.nets['power'].res_gen['p_mw']),
        "Total and resultant load": sum(multinet.nets['power'].res_load['p_mw']),
        "External grid": multinet.nets['power'].res_ext_grid['p_mw'][0],
        "Losses": sum(multinet.nets['power'].res_line['pl_mw']),
        "Total Ele load": sum(multinet.nets['power'].res_load['p_mw'][:14]),
        "Total P2G load": sum(multinet.nets['power'].res_load['p_mw'][14:]),

        
        # "Hydrogen demand (kg/s)": sum(multinet.nets['hydrogen'].res_sink['mdot_kg_per_s']),
        # "Hydrogen external (kg/s)": sum(multinet.nets['hydrogen'].res_ext_grid['mdot_kg_per_s'])
        

        "Hydrogen demand": sum(multinet.nets['hydrogen'].res_sink['mdot_kg_per_s'])*(energy_hydrogen*3600)/1e3,
        "Hydrogen supply": multinet.nets['hydrogen'].res_source['mdot_kg_per_s'].sum(skipna=True)*(energy_hydrogen*3600)/1e3,
        "Hydrogen external": sum(multinet.nets['hydrogen'].res_ext_grid['mdot_kg_per_s'])*(energy_hydrogen*3600)/1e3
    }
        
    
    from IPython.display import display
    display(results)
    
    print("\nSelected Scenarios:")
    for key, value in Scenarios.items():
        print(f"{key}: {value}")
        
        
