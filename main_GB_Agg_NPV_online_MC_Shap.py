# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 03:48:57 2025

@author: Mhdella
"""



import numpy as np
import pandas as pd
import os
import time
import pyomo.environ as pyoen
from pyomo.environ import *
import pyomo.mpec as pyompec
from pyomo.opt import SolverFactory
from itertools import combinations
from math import factorial
from tqdm import tqdm

from agg_results_handler import *
from multinet_GB_Agg import *


import warnings
warnings.filterwarnings('ignore')  # Ignores all warnings

start_time = time.time()


def data_import(Scenarios):
    
    GBdata_path = "GB_2050_Data/"  # Path to the Data folder

    Players_data = pd.read_csv(GBdata_path + 'GB_market_players.csv')
    Economics_data = pd.read_excel(GBdata_path + 'GB_Economic_Parameters.xlsx', sheet_name='Econ_Players', index_col = 0)
    GB_Generations_Profiles = pd.read_excel(GBdata_path + 'GB_Generations.xlsx', index_col = 0)
    Gens_profile = GB_Generations_Profiles.iloc[Scenarios['time_steps']].reset_index()
    Gens_profile.columns = ['type', 'scale']
    
    
    # if Scenarios['Demand_level']=='Peak':
    #     Gens_profile['scale'] = 1  ## For OPGF with all installled capacity of generations
    #     # Gens_profile['scale'] = Gens_profile['scale']

    
    Players_economics = pd.concat([Economics_data] * 17, ignore_index=True)
    # Players_economics[['id', 'zone_id', 'max_p_mw']] = Players_data[['id', 'zone_id', 'max_p_mw']]
    Players_economics[['id', 'zone_id']] = Players_data[['id', 'zone_id']]


    Players_economics = Players_economics.merge(Gens_profile[['type', 'scale']], on='type', how='left')
    Players_economics['max_p_mw'] = Players_economics['max_p_mw'] * Players_economics['scale']

    
    systemData = {
        
        'Year': int(year), 'scenario': scenario,
        'Players':  Players_economics,
        'Economics': Economics_data,
        'Economics_H2': pd.read_excel(GBdata_path + 'GB_Economic_Parameters.xlsx', sheet_name='Econ_H2', index_col = 0),

        'Carbon Price'      : pd.read_excel(GBdata_path + 'carbon_price.xlsx'), # carbon price ,
        
        # 'Installed capacity': pd.read_excel(GBdata_path + 'installed_capacity.xlsx', sheet_name=scenario, index_col = 0), #installed capacity
        'Installed capacity': pd.read_excel(GBdata_path + 'Agg_installed_capacity.xlsx', sheet_name=scenario, index_col = 0), #installed capacity

        
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
        'G2G Data': pd.read_excel(GBdata_path + 'GB_G2G.xlsx', sheet_name=None),  
        'G2P Data': pd.read_excel(GBdata_path + 'GB_G2P.xlsx', sheet_name=None), 
    }
    
    
    # Process P & Q (demand) for Bus data (Q is not needed, because DC power flow)
    systemData['Bus Data']['2']=systemData['Zone_Demand'][Scenarios['Zone_Demand']].values # P(demand)
    # systemData['Bus Data']['3']= 0.3 * systemData['Zone_Demand'][Scenarios['Zone_Demand']].values  # Q(demand)
        
    systemData['Gen Data']['1'] = 0
    systemData['Gen Data']['8'] = Players_economics['max_p_mw']    

    storage_mask = Players_economics['type'] == 'Storage'
    systemData['Gen Data'].loc[storage_mask, '9'] = Players_economics.loc[storage_mask, 'max_p_mw']
    
   
    # Normalize the values by dividing each column by its sum
    systemData['GB Demand Profiles'] = systemData['GB Demand Profiles'].div(systemData['GB Demand Profiles'].sum(axis=0), axis=1)*1e6
                                                         
    
    # Process time series profile
    systemData['Time Series Profiles'] = {
        'e_load_time_series': systemData['GB Demand Profiles'][Scenarios['Profile']].values,
        
        'g_load_time_series': systemData['GB Demand Profiles']['H2 industry'].values,
                                          }                                         
        
   
    systemData['Power Demand'] = systemData['Bus Data']['2'],
    # systemData['Gas Consumption'] = pd.read_excel(path + 'gasConsumption_heating_input/gasCons_'+ scenario + '.xlsx', sheet_name = year, index_col = 0),
    
    

    systemData['Scenarios_id'] = Scenarios
        
    # # Filter out players with zero capacity (2050)
    systemData['Players_2050'] = systemData['Players'][systemData['Players']["max_p_mw"] > 0]
    
    multinet_0 = intital_multinet(time_steps,  systemData)

    systemData['Q_e'] = sum(multinet_0.nets['power'].load['p_mw'][:14])
    

    
    systemData['Players_H2'] = pd.read_excel(GBdata_path + 'GB_Economic_Parameters.xlsx', sheet_name='Econ_Players_H2', index_col = 0)
    
    systemData['Demand_h'] = systemData['H2_Demand'][Scenarios['H2_Demand']]*1e6/8760,

    systemData['Q_h'] = np.sum(systemData['Demand_h'])
    
    systemData['H2_P2G'] = sum(multinet_0.nets['power'].load['p_mw'][14:65])
    
    systemData['max_p_mw_updated'] = 'Not-Updated'
    
    systemData['Opex_Costs'] = pd.read_excel(GBdata_path + 'GB_Economic_Parameters.xlsx', sheet_name='Cost_Calc', index_col = 0)

    systemData['Installed capacity'][2050] = systemData['Installed capacity'][gens_fg] 

    
    return systemData




def variable_list(model, year, data):
    
    data = systemData

    global ub
    ub = data['Installed capacity'][int(year)] - data['Installed capacity'][2025]

        
    global selected_players

    selected_players = get_selected_players('type') # Type of players
    # selected_players = get_selected_players('ele_players') # Only the elec  players
    
    
    # Initialize the model's set with the selected players
    model.p = pyoen.Set(initialize=selected_players)
   
    
    # # Initialize model.q with 2050 installed capacities
    
    # model.q = Var(model.p, within=NonNegativeReals, bounds=(0, 300000), initialize=1000)
    
    capacity_dict = {i: data['Installed capacity'][2050][i] for i in selected_players}
    model.q = Var(model.p, within=NonNegativeReals, bounds=(0, 300000), 
                  initialize=lambda model, i: capacity_dict[i])
    
    model.I = Var(model.p, within=NonNegativeReals, initialize=0)
    model.sigma = Var(model.p, within=NonNegativeReals, initialize=0)
    model.delta = Var(model.p, within=NonNegativeReals, initialize=0)
    model.alpha = Var(model.p, initialize=0)
   
    
    return model





def variable_list(model, year, data):
    
    data = systemData

    global ub
    ub = data['Installed capacity'][int(year)] - data['Installed capacity'][2025]

        
    global selected_players

    selected_players = get_selected_players('type') # Type of players
    # selected_players = get_selected_players('ele_players') # Only the elec  players
    
    
    # Initialize the model's set with the selected players
    model.p = pyoen.Set(initialize=selected_players)
   
    
    # # Initialize model.q with 2050 installed capacities
    
    # model.q = Var(model.p, within=NonNegativeReals, bounds=(0, 300000), initialize=1000)
    
    capacity_dict = {i: data['Installed capacity'][2050][i] for i in selected_players}
    model.q = Var(model.p, within=NonNegativeReals, bounds=(0, 300000), 
                  initialize=lambda model, i: capacity_dict[i])
    
    model.I = Var(model.p, within=NonNegativeReals, initialize=0)
    model.sigma = Var(model.p, within=NonNegativeReals, initialize=0)
    model.delta = Var(model.p, within=NonNegativeReals, initialize=0)
    model.alpha = Var(model.p, initialize=0)
   
    
    return model




def parameter_list(model, year, systemData):
    data = systemData
    # Parameter list ----------------------------------------------------------
    
    model.years = pyoen.Param(initialize=int(year) - 2025)  # value in 'years'

    # Filter the relevant parameters for selected players
    filtered_costs = {i: data['Agg_Players']['costs'][i] for i in model.p if i in data['Agg_Players']['costs']}
    filtered_costs_mwyr = {i: data['Agg_Players']['opex_mw_year'][i] for i in model.p if i in data['Agg_Players']['opex_mw_year']}
    filtered_CF = {i: data['Agg_Players']['CF'][i] for i in model.p if i in data['Agg_Players']['CF']}
    filtered_vector = {i: data['Agg_Players']['final_vector'][i] for i in model.p if i in data['Agg_Players']['final_vector']}

    filtered_epsilon = {i: data['Agg_Players']['epsilon'][i] for i in model.p if i in data['Agg_Players']['epsilon']}
    filtered_phi = {i: data['Agg_Players']['phi'][i] for i in model.p if i in data['Agg_Players']['phi']}
    filtered_lamda = {i: data['Agg_Players']['economic_life'][i] for i in model.p if i in data['Agg_Players']['economic_life']}

    filtered_k = {i: data['Installed capacity'][2025][i] for i in model.p if i in data['Installed capacity'][2025]}
    # filtered_k = {i: abs(data['Agg_Players']['max_p_mw'])[i] for i in model.p if i in data['Agg_Players']['max_p_mw']}
    # filtered_k = {i: data['Installed capacity'][2050][i] for i in model.p if i in data['Installed capacity'][2050]}

    filtered_emissions = {i: data['Agg_Players']['emissions'][i] for i in model.p if i in data['Agg_Players']['emissions']}
    filtered_discount_rate = {i: data['Agg_Players']['discount_rate'][i] for i in model.p if i in data['Agg_Players']['discount_rate']}

    filtered_installed_capacity = {
        i: (data['Installed capacity'][int(year)][i]) - (data['Installed capacity'][2025][i])
        for i in model.p if i in data['Installed capacity'][int(year)]}

    # Initialize the model parameters
    model.c = pyoen.Param(model.p, initialize=filtered_costs)  
    model.cmwyr = pyoen.Param(model.p, initialize=filtered_costs_mwyr)  
    model.CF = pyoen.Param(model.p, initialize=filtered_CF) 
    
    model.vector = pyoen.Param(model.p, initialize=filtered_vector, within=Any)

    model.eps = pyoen.Param(model.p, initialize=filtered_epsilon)  
    
    
    if phi_fg == 0.0:
        model.phi = pyoen.Param(model.p, initialize=filtered_phi)
    elif phi_fg != 0.0:
        model.phi = pyoen.Param(model.p, initialize = phi_fg)   
    

    # model.lamda = pyoen.Param(model.p, initialize=filtered_lamda)  
    model.lamda = pyoen.Param(model.p, initialize=invest_span) 
    model.k = pyoen.Param(model.p, initialize=filtered_k)  
    model.emis = pyoen.Param(model.p, initialize=filtered_emissions)  
    model.iMAX = pyoen.Param(model.p, initialize=filtered_installed_capacity)  
    model.rate = pyoen.Param(model.p, initialize=filtered_discount_rate)  
    
    model.AF = pyoen.Param(model.p, initialize=lambda model, i:
                           ((1 + model.rate[i]) ** model.years - 1) /
                           (model.rate[i] * (1 + model.rate[i]) ** model.years))

    model.Q_e = systemData['Q_e']
    model.Q_h = systemData['Q_h']
    
    
    model.pE = pyoen.Param(initialize=(price_list[0]))  
    model.pG = pyoen.Param(initialize=(price_list[1]))
    model.pH = pyoen.Param(initialize=(price_list[2]))
    

    CO2_penalty = systemData['Carbon Price'].loc[systemData['Carbon Price']['Year'] == int(year), scenario].iloc[0]
    model.CO2 = pyoen.Param(initialize=(CO2_penalty))  #£/tonne CO2
    
    # model.CO2 = pyoen.Param(initialize=50)  #£/tonne CO2

    return model





def calculate_shapley_values(model, num_samples=1000, debug_coalitions=None):
    
    import random
    from random import randint, sample
    
    global systemData, Players_option, GBdata_path
    
    # Extract players from the model
    players = list(model.p)
    n = len(players)
    
    # Coalition profits calculator (realistic)
    def coalition_profits(coalition):
        if not coalition:
            return 0
    
        coalition_npv_profit = 0
        for player in coalition:
            # Get parameters
            capex = model.eps[player]               # £/MW
            max_mw = model.q[player].value  # Installed capacity in MW
            life = model.lamda[player]              # years
            r = model.rate[player]                  # discount rate
            cf = model.CF[player]                   # capacity factor
            emissions = model.emis[player]          # tCO2/MWh
            phi = model.phi[player]                 # fixed cost coefficient
            energy_type = model.vector[player]      # Electricity or Hydrogen
    
            price = model.pE.value if energy_type == "Electricity" else model.pH.value
            annual_output = max_mw * cf * 8760
            revenue = annual_output * price
            op_expense = annual_output * model.c[player]  # cmwyr is Op cost £/MWh
            carbon_cost = annual_output * emissions * model.CO2.value
            fixed_cost = phi * capex * max_mw  # £
            invest = capex * max_mw  # £
    
            # Net investment: straight-line depreciation assumed, so this simplifies
            net_investment = invest * (1 - ((life - life) / (life * (1 + r) ** life)))
            # Simplifies to:
            net_investment = invest
    
            # Annuity factor (AF)
            af = (1 - (1 + r) ** -life) / r
    
            # Operational net benefit
            operational_net = revenue - op_expense - carbon_cost
    
            # NPV calculation
            npv = af * (operational_net - fixed_cost) - net_investment
    
            # Optional debug output
            if debug_coalitions and frozenset(coalition) in debug_coalitions:
                print(f"Player {player}: Type={energy_type}, Revenue={revenue}, OpCost={op_expense}, "
                      f"CarbonCost={carbon_cost}, FixedCost={fixed_cost}, Invest={invest}, NPV={npv}")
    
            coalition_npv_profit += npv
    
        if debug_coalitions and frozenset(coalition) in debug_coalitions:
            print(f"Coalition {list(coalition)}: Total NPV={coalition_npv_profit}")
            
        return coalition_npv_profit
    
    
    # Monte Carlo approximation for Shapley values
    shapley_values_profits = {player: 0 for player in players}
    total_iterations = n * num_samples
    
    with tqdm(total=total_iterations, desc="Computing Shapley values (Monte Carlo)") as progress_bar:
        for player in players:
            other_players = [p for p in players if p != player]
            for _ in range(num_samples):
                subset_size = randint(0, n - 1)  # Random coalition size excluding player
                subset = set(random.sample(other_players, subset_size))  # Random subset
                with_player = subset | {player}
                profits_without_player = coalition_profits(subset)
                profits_with_player = coalition_profits(with_player)
                marginal_contribution = profits_with_player - profits_without_player
                shapley_values_profits[player] += marginal_contribution / num_samples
                progress_bar.update(1)
    
    # Save Shapley values
    global df_shap
    df_shap = pd.DataFrame([
        (player, systemData['Agg_Players'].loc[player, 'type'], shap_value)
        for player, shap_value in shapley_values_profits.items()
    ], columns=['Player', 'Type', 'Shapley Value'])
    
    filename = f"shaps_MCApprox_NPV7_Agg_GB_{Players_option}.csv"
    file_path = os.path.join('Output', filename)
    df_shap.to_csv(file_path, index=False)
    
    if 'shap_values' not in systemData or not isinstance(systemData['shap_values'], list) or len(systemData['shap_values']) == 0:
        systemData['shap_values'] = [None]
    systemData['shap_values'][0] = df_shap
    
    # Store Shapley values in model
    for player in players:
        model.Shapley_profits[player] = shapley_values_profits[player]
    
    # Report results
    print(f"Shapley values (profits) saved to {file_path}")
    
    return shapley_values_profits




def distribute_generation_by_shapley(model, systemData):
    
    global pos_shap_elec, neg_shap_elec, pos_shap_hydrogen, neg_shap_hydrogen
    # Reset all q values to 0
    for p in model.p:
        model.q[p].set_value(0)

    # Match Shapley values with players
    shapley_values = match_shap_values_with_installed_capacity(systemData)

    # Separate players by type
    electric_players = [p for p in shapley_values if p not in [24, 25]]  # Exclude p2g (24) and g2p (25)
    hydrogen_players = [24, 25]  # Only p2g and g2p for hydrogen demand

    # Separate positive and negative Shapley values for both groups
    pos_shap_elec = [p for p in electric_players if shapley_values[p] > 0]
    neg_shap_elec = [p for p in electric_players if shapley_values[p] < 0]
    
    pos_shap_hydrogen = [p for p in hydrogen_players if shapley_values[p] > 0]
    neg_shap_hydrogen = [p for p in hydrogen_players if shapley_values[p] < 0]

    # Total demands (electric and hydrogen)
    total_electric_demand = model.Q_e  # Electric demand in GWh
    total_hydrogen_demand = model.Q_h  # Hydrogen demand in GWh

    # Start by allocating electric demand
    allocate_demand(model, pos_shap_elec, neg_shap_elec, total_electric_demand, shapley_values, 'electric')
    
    # Allocate hydrogen demand
    allocate_demand(model, pos_shap_hydrogen, neg_shap_hydrogen, total_hydrogen_demand, shapley_values, 'hydrogen')

    # Print results
    total_allocated_gen = sum(model.q[p].value for p in model.p)
    print(f"Total Allocated Generation: {round(total_allocated_gen, 2)} MWh")
    if total_electric_demand + total_hydrogen_demand - total_allocated_gen > 1e-3:
        print(f"WARNING: {round((total_electric_demand + total_hydrogen_demand) - total_allocated_gen, 2)} MWh unmet (capacity or policy constraints)")
    else:
        print("All demand met.")
        

    return model




def allocate_demand(model, pos_shap_players, neg_shap_players, total_demand, shapley_values, demand_type):
    
    MAX_GEN_PER_PLAYER = model.iMAX.extract_values()

    # Track remaining demand
    demand_remaining = total_demand
    total_gen = 0

    # --- Phase 1: Allocate using positive Shapley value players ---
    if len(pos_shap_players) > 0:
        pos_total_shap = sum(shapley_values[p] for p in pos_shap_players)
        for p in pos_shap_players:
            weight = shapley_values[p] / pos_total_shap
            gen = min(weight * total_demand, MAX_GEN_PER_PLAYER[p])
            if gen > demand_remaining:
                gen = demand_remaining
            model.q[p].value = gen
            demand_remaining -= gen
            total_gen += gen
            
            if demand_remaining <= 0:
                break

    # --- Phase 2: Allocate remaining demand to negative Shapley players (inverse of absolute Shapley value) ---
    if demand_remaining > 0 and len(neg_shap_players) > 0:
        
        # Inverse of absolute Shapley values (the least negative gets priority)
        neg_shap_inv = {p: 1 / abs(shapley_values[p]) for p in neg_shap_players}
        total_inv_shap = sum(neg_shap_inv.values())

        # Proportional allocation based on inverse of Shapley values
        for p in neg_shap_players:
            weight = neg_shap_inv[p] / total_inv_shap
            gen = min(weight * demand_remaining, MAX_GEN_PER_PLAYER[p])
            model.q[p].value = gen
            demand_remaining -= gen
            total_gen += gen

            if demand_remaining <= 0:
                break

    # --- Phase 3: Redistribute remaining demand among negative Shapley players (if applicable) ---
    if demand_remaining > 0 and len(neg_shap_players) > 0:
        # Players with remaining capacity
        neg_cap_remaining = {
            p: MAX_GEN_PER_PLAYER[p] - model.q[p].value
            for p in neg_shap_players if model.q[p].value < MAX_GEN_PER_PLAYER[p]
        }

        # If there are negative players with remaining capacity, redistribute demand
        if neg_cap_remaining:
            neg_weights = np.array([neg_shap_inv[p] for p in neg_cap_remaining])
            neg_weights /= neg_weights.sum()
            additional_alloc = neg_weights * demand_remaining

            for p, extra in zip(neg_cap_remaining.keys(), additional_alloc):
                model.q[p].value += min(extra, MAX_GEN_PER_PLAYER[p] - model.q[p].value)

    # Print summary for each demand type
    print(f"Allocated {demand_type} demand: {round(total_gen, 2)} MWh, Remaining: {round(demand_remaining, 2)} MWh")



def calculate_policy_opex_support(model, systemData, neg_shaps_list, demand_type='electric'):
    
    global policy_support, dic_supports_e, dic_supports_h, allocated_group
    
    allocated_group = [i for i in neg_shaps_list if model.q[i].value > 1e-3]

    if demand_type=='electric':
        market_price = model.pE.value
    elif demand_type=='hydrogen':
        market_price = model.pH.value
        
    dic_supports = {}
    policy_support = 0.0
    
    for p in allocated_group:
        Cost_opex = model.c[p]   # £/MWh 
        Cost_co2 = model.emis[p] * model.CO2.value   # £/MWh
        Cost = Cost_opex + Cost_co2
        
        loss_cost = max(0, Cost - market_price)
        generated = model.q[p].value  # MWh
        
        # The support amount (£) for player
        policy_support += loss_cost * generated
        
        dic_supports[p] = loss_cost * generated
    
    if demand_type=='electric':
        dic_supports_e = dic_supports
    elif demand_type=='hydrogen':
        dic_supports_h = dic_supports
        
    return policy_support



def calculate_policy_npv_support(model, systemData, neg_shaps_list, demand_type='electric'):
    
    global policy_support_npv, dic_supports_npv_e, dic_supports_npv_h, allocated_group

    allocated_group = [i for i in neg_shaps_list if model.q[i].value > 1e-3]
    
    if demand_type == 'electric':
        market_price = model.pE.value
    elif demand_type == 'hydrogen':
        market_price = model.pH.value

    dic_supports = {}
    policy_support_npv = 0.0

    for p in allocated_group:

        q = model.q[p].value
        I = model.I[p].value
        k = model.k[p]
        c = model.c[p]
        eps = model.eps[p]
        phi = model.phi[p]
        lamda = model.lamda[p]
        rate = model.rate[p]
        emis = model.emis[p]
        CO2 = model.CO2.value
        AF = model.AF[p]

        revenue = q * market_price
        
        opex = q * c

        co2_cost = q * emis * CO2

        # FC = phi * eps * (k + I)
        FC = phi * eps * q

        # NI = eps * I * (1 - ((lamda - lamda) / (lamda * (1 + rate) ** lamda)))  # simplifies to eps * I
        NI = eps * q * (1 - ((lamda - lamda) / (lamda * (1 + rate) ** lamda)))  # simplifies to eps * I

        npv = AF * (revenue - opex - co2_cost - FC) - NI

        # If NPV < 0, support is needed
        support_amount = max(0, -npv)

        policy_support_npv += support_amount
        # policy_support_npv += FC

        dic_supports[p] = support_amount
        
        # print(f"Player {p}")
        # print(f"  q = {model.q[p].value}")
        # print(f"  I = {model.I[p].value}, k = {model.k[p]}")
        # print(f"  eps = {model.eps[p]}, phi = {model.phi[p]}")
        # print(f"  revenue = {q * market_price}")
        # print(f"  opex = {q * model.c[p]}")
        # print(f"  co2_cost = {q * model.emis[p] * model.CO2.value}")
        # print(f"  FC = {phi * eps * (k + I)}")
        # print(f"  NI = {eps * I}")
        # print(f"  AF = {model.AF[p]}")
        # print("----------")

    if demand_type == 'electric':
        dic_supports_npv_e = dic_supports
    elif demand_type == 'hydrogen':
        dic_supports_npv_h = dic_supports
        

    return policy_support_npv




    

def match_shap_values_with_installed_capacity(systemData):
    
    global shap_df
    shap_data = systemData['shap_values'][0]  # Extract the first element (dictionary)
    
    # shap_df = shap_data['Sheet']
    shap_df = shap_data
    
    systemData['Agg_Players']['max_p_mw'] = abs(systemData['Agg_Players']['max_p_mw'])
    selected_players_df = systemData['Agg_Players'].iloc[selected_players]

    capacity_per_type = abs(selected_players_df.groupby('type')['max_p_mw'].sum())
    shap_values_per_type = shap_df.set_index('Type')['Shapley Value'].to_dict()

    shapley_values = {}

    # Distribute SHAP values based on type and capacity proportion
    for _, row in selected_players_df.iterrows():
        player_id = row['id']
        player_type = row['type']
        player_capacity = row['max_p_mw']

        # Get the total SHAP value for the player's type
        if player_type in shap_values_per_type and player_type in capacity_per_type:
            total_shap_value = shap_values_per_type[player_type]
            total_capacity = capacity_per_type[player_type]

            # Distribute SHAP value proportionally
            shapley_values[player_id] = (player_capacity / total_capacity) * total_shap_value
        else:
            # Assign 0 if the type is not in shap_values
            shapley_values[player_id] = 0
            

    return shapley_values




    
def cooperative_game(year, scenario, data):
        
    global shapley_values
        
    model = pyoen.ConcreteModel()
    
    variable_list(model, year, data)
        
    parameter_list(model, year, data)
    
    model.Shapley_profits = pyoen.Param(model.p, mutable=True)
        
    
    shapley_values = calculate_shapley_values(model)

    

    distribute_generation_by_shapley(model, systemData)
    
    

    return model





def initial_run_OPGF(model, systemData):
    
    genCapacities = model.q.extract_values()
    genCapacities = pd.DataFrame.from_dict(genCapacities, orient='index')
    
    updated_gen_data, updated_p2g_data, updated_g2g_data, updated_g2p_data = distribute_players_capacities(genCapacities, systemData)
    
    updated_gen_data['out_pmw'] = updated_gen_data['out_pmw'] * systemData['Players'].scale
    
    
    gen_scale = (model.Q_e + model.Q_h) / updated_gen_data['out_pmw'].sum()
    
    # systemData['Gen Data']['8'] = updated_gen_data['out_pmw']
    systemData['Gen Data']['8'] = updated_gen_data['out_pmw']*gen_scale

    

    storage_mask = updated_gen_data['type'] == 'Storage'
    # systemData['Gen Data'].loc[storage_mask, '9'] = -abs(updated_gen_data.loc[storage_mask, 'out_pmw'])
    systemData['Gen Data'].loc[storage_mask, '9'] = abs(updated_gen_data.loc[storage_mask, 'out_pmw'])

    
    systemData['P2G Data'] = updated_p2g_data
    
    systemData['G2G Data'] = updated_g2g_data
    
    systemData['G2P Data'] = updated_g2p_data


    # Pick an hour for the model to run
    hour_of_day = time_steps
    
    multinet = run_OPGF(hour_of_day, systemData['Gen Data'], systemData)  # Running the OPGF
    
    return multinet





def cost_check(model, multinet, systemData):
    
    players = systemData['Players']
    
    H2_types = {"g2p(H2-CCGT)", "g2p(H2-OCGT)", "g2p(Fuel Cell)", "p2g", "g2g"}
    H2_plyaers_ids = systemData['Agg_Players'][systemData['Agg_Players']["type"].isin(H2_types)].index.tolist()
    systemData['H2_plyaers_ids'] = H2_plyaers_ids 
    
    
    H2_plyaers = systemData['H2_plyaers_ids']
    
    ele_players = [player for player in selected_players if player not in H2_plyaers]

    cost_GT = sum(pyoen.value(model.q[tech] * model.c[tech]) for tech in ele_players)
    
    cost_OPGF = sum(multinet['nets']['power']['res_gen']['p_mw']* players['costs'])
        
    # CO2 emissions cost for GT Model
    emission_GT = sum(pyoen.value(model.q[tech] * model.emis[tech]) for tech in ele_players)
    co2_cost_GT = emission_GT * pyoen.value(model.CO2)

    # CO2 emissions cost for OPGF Model
    emission_OPGF = sum(multinet['nets']['power']['res_gen']['p_mw'] * players['emissions'])
    co2_cost_OPGF = emission_OPGF * pyoen.value(model.CO2)

    # CO2 emissions cost for G2G-CCS 
    emission_G2G= sum(multinet.nets['gas'].sink['mdot_kg_per_s']) * (14.64 * 3600) / 1000
    CO2_cost_G2G = emission_G2G * pyoen.value(model.CO2) * 0.022

    g2g_out = multinet.nets['hydrogen'].res_source['mdot_kg_per_s'][51:].sum(skipna=True) * (39.41 * 3600) / 1000
    g2g_in = multinet.nets['gas'].res_sink['mdot_kg_per_s'][:34].sum(skipna=True)*(energy_gas*3600)/1000
    
    c_g2g = systemData['Players_H2'].loc[systemData['Players_H2']['type'] == 'g2g', 'costs'].values[0]
    
    g2gCost= g2g_in * c_g2g + CO2_cost_G2G

    cost_GT = cost_GT + co2_cost_GT + g2gCost
    cost_OPGF = cost_OPGF + co2_cost_OPGF + g2gCost 
    
    return {
        "cost_GT": cost_GT,
        "cost_OPGF": cost_OPGF,
        "emission_GT": emission_GT,
        "co2_cost_GT": co2_cost_GT,
        "emission_OPGF": emission_OPGF,
        "co2_cost_OPGF": co2_cost_OPGF
            }





def update_GT(year, scenario, multinet, systemData):
    
    systemData['Agg_Players'] = update_agg_players_max_p_mw(multinet, systemData, selected_players)
    # systemData['Agg_Players']['max_p_mw'] = systemData['Agg_Players']['max_p_mw']*1.5

    
    systemData['Q_e'] = sum(multinet.nets['power'].load['p_mw'])
    # systemData['Q_e'] = sum(multinet.nets['power'].load['p_mw'][:14])
    
    from agg_results_handler import OPGF_results
    opf_results = OPGF_results(multinet, systemData)
    

    systemData['Q_h'] = opf_results['Total H2 Demand']

    systemData['Q_g'] = opf_results['G2G Gas Demand']
    
    
    model = cooperative_game(year, scenario, systemData)
    
        
    return model





def get_selected_players(selection_option):
    global Players_option
    
    if systemData['max_p_mw_updated'] == 'Not-Updated':
        
        systemData['Players']['Efficiency'] = 1
        
        systemData['Players_H2'] = systemData['Players_H2'].copy()  
        systemData['Players_H2'][['id', 'zone_id', 'scale']] = 1 
        
        systemData['Agg_Players'] = pd.concat([systemData['Players'][:21], systemData['Players_H2']], ignore_index=True)
        
        systemData['Agg_Players'].reset_index(drop=True, inplace=True)
        systemData['Agg_Players']['id'] = systemData['Agg_Players'].index  
    
        systemData['Agg_Players']['zone_id'] = 0
        
    
        # systemData['Agg_Players']['max_p_mw'] = systemData['Installed capacity'][int(year)] 
        systemData['Agg_Players']['max_p_mw'] = systemData['Installed capacity'][int(year)] * systemData['Agg_Players'].scale
       
        systemData['max_p_mw_updated'] = 'Updated'  # Mark as updated
    
    
    if selection_option == 'ele_players':
        systemData['Agg_Players'] = systemData['Agg_Players'][:21]  ## Elec Players only
    

    # One player per unique type (excluding types where max_p_mw == 0)
    selected_types = systemData['Agg_Players']['type'].unique()
    players_ids = systemData['Agg_Players'].loc[systemData['Agg_Players']['type'].isin(selected_types)].drop_duplicates(subset=['type'])[['type', 'id']]
    players_list = players_ids['id'].tolist()
        
    filtered_players_mask = abs(systemData['Agg_Players'].loc[players_list, 'max_p_mw']) > 0
    unique_type_players = systemData['Agg_Players'].loc[players_list][filtered_players_mask]['id'].tolist()    
    

    if selection_option == 'type':
        Players_option = 'type_players'
    elif selection_option == 'ele_players':
         Players_option = 'ele_players'
    else:
        raise ValueError("Invalid selection_option. Choose from 'capacity', 'zone', 'all', 'top3', or 'type'.")

    selected_players = unique_type_players

    # Store selected players in systemData dictionary
    systemData['selected_players'] = selected_players
    
    # # # Set 'max_p_mw' to 0 for players not in selected list
    systemData['Agg_Players'].loc[~systemData['Agg_Players']['id'].isin(selected_players), 'max_p_mw'] = 0
    
    # systemData['Agg_Players'].loc[~systemData['Agg_Players']['id'].isin(selected_players), 'max_p_mw'] = 10
    
    return selected_players




def distribute_players_capacities(genCapacities, systemData):
    
    H2_types = {"g2p(H2-CCGT)", "g2p(H2-OCGT)", "g2p(Fuel Cell)", "p2g", "g2g"}
    H2_plyaers_ids = systemData['Agg_Players'][systemData['Agg_Players']["type"].isin(H2_types)].index.tolist()
    systemData['H2_plyaers_ids'] = H2_plyaers_ids 
    
    # systemData['Players'][['max_p_mw', 'type', 'scale']].iloc[selected_players]
    filtered_data = systemData['Agg_Players'][['max_p_mw', 'type', 'scale']].iloc[selected_players]
    filtered_data['model_q'] = np.array(genCapacities).reshape(-1) * np.where(filtered_data['scale'] < 0, -1, 1)
    
    # filtered_data['model_q'] = np.array(genCapacities).reshape(-1) * filtered_data['scale'] 

    storage = filtered_data[filtered_data['type'] == 'Storage'][filtered_data['model_q'] < 0]
    storage_value = storage['model_q'].sum()
    clean_energy_sources = filtered_data[filtered_data['type'].isin(['Offshore Wind', 'PV', 'Nuclear'])]
    num_clean_sources = clean_energy_sources.shape[0]
    distributed_value = 3 * abs(storage_value) / num_clean_sources
    filtered_data['planned_capacity'] = 0
    
    if storage_value <0:
        filtered_data.loc[clean_energy_sources.index, 'planned_capacity'] =  distributed_value
    filtered_data['model_q'] = filtered_data['model_q'] + filtered_data['planned_capacity'] 


    type_to_model_q = filtered_data.groupby('type')['model_q'].sum().to_dict()
    id_to_type = systemData['Players']['type'].to_dict()
    
    genData = systemData['Gen Data'] 
    
    gen_data_types = genData.index.map(id_to_type)
    type_to_sum_gendata = genData.groupby(gen_data_types)['8'].sum().to_dict()
    
    updated_gen_data = genData.copy()
    updated_gen_data['type'] = updated_gen_data.index.map(id_to_type)
    updated_gen_data['model_q'] = updated_gen_data['type'].map(type_to_model_q)
    updated_gen_data['model_q'].fillna(0, inplace=True)
    updated_gen_data['aggreg'] = updated_gen_data['type'].map(type_to_sum_gendata)
    updated_gen_data['out_pmw'] = updated_gen_data['model_q'] * (genData['8'] / updated_gen_data['aggreg'])
    updated_gen_data['out_pmw'].fillna(0, inplace=True)
    
    ###################################
    
    import copy
    
    updated_p2g_data = copy.deepcopy(systemData['P2G Data'])
    
    p2g_value_gw = filtered_data.loc[filtered_data['type'] == 'p2g', 'model_q'].values[0] / 1000  

    technology_totals = {tech: df['Capacity(GW)'].sum() for tech, df in updated_p2g_data.items()}
    
    for tech, df in updated_p2g_data.items():
        if technology_totals[tech] > 0:  
            df['Capacity(GW)'] = df['Capacity(GW)'] / technology_totals[tech] * p2g_value_gw/3


    ###################################

    updated_g2g_data = copy.deepcopy(systemData['G2G Data'])
    
    g2g_value_gw = filtered_data.loc[filtered_data['type'] == 'g2g', 'model_q'].values[0] / 1000  
    
    technology_totals = {tech: df['Capacity(GW)'].sum() for tech, df in updated_g2g_data.items()}
    
    for tech, df in updated_g2g_data.items():
        if technology_totals[tech] > 0:  
            df['Capacity(GW)'] = df['Capacity(GW)'] / technology_totals[tech] * g2g_value_gw
        
    ###################################
    
    updated_g2p_data = copy.deepcopy(systemData['G2P Data'])
    
    g2p_values = {
        'FC': filtered_data.loc[filtered_data['type'] == 'g2p(Fuel Cell)', 'model_q'].values[0] / 1000,
        'H2-CCGT': filtered_data.loc[filtered_data['type'] == 'g2p(H2-CCGT)', 'model_q'].values[0] / 1000,
        'H2-OCGT': filtered_data.loc[filtered_data['type'] == 'g2p(H2-OCGT)', 'model_q'].values[0] / 1000
    }
    
    technology_totals = {tech: df['Capacity(GW)'].sum() for tech, df in updated_g2p_data.items()}
    
    for tech, df in updated_g2p_data.items():
        if technology_totals[tech] > 0:  
            df['Capacity(GW)'] = df['Capacity(GW)'] / technology_totals[tech] * g2p_values[tech]


    
    return updated_gen_data, updated_p2g_data, updated_g2g_data, updated_g2p_data




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
    
    return systemData['Agg_Players']


        
    

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

    # # scenario_options = [H2 option, Zone option, and Profile option]
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

#################################################

    years = ['2025','2030','2035','2040','2045','2050']
    scenarios = ['FS', 'CT', 'LW', 'ST']
    
    year = years[5]
    scenario = scenarios[2]
    
#################################################
    
    # Set globals
    global phi_fg, gens_fg, invest_span, price_list
    
    
    phi_fg = 0.0  ## Set phi = Variable = Fixed O&M/Capex
    
    # phi_fg = 0.10
    # phi_fg = 0.25
    # phi_fg = 0.50
    # phi_fg = 0.75
    # phi_fg = 1
    
    
    # invest_span = 5
    # invest_span = 10
    # invest_span = 15
    # invest_span = 20
    invest_span = 25
    
    
    price_level = 'high';  price_list = [100, 75, 150] # Elec, Gas, Hydrogen
    # price_level = 'low';  price_list = [40, 15, 50]


#################################################    
        
    
    
    
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
    
    for gens_flag, scenario_name in plan_scenarios.items():
        print(f"\nRunning scenario: {scenario_name} ({gens_flag})")
        
        gens_fg = gens_flag  # Set scenario flag
        
        #################################################

        # time_steps = 0
        # time_steps = 1
        # time_steps = 2
        # time_steps = 3
        time_steps = 13
        # time_steps = 14
        # time_steps = 19
        # time_steps = 23


        Scenarios['time_steps'] = time_steps
        
        # Scenarios['Demand_level'] = 'Normal'
        
        Scenarios['Demand_level'] = 'Peak'  ## This is for long-term planning

        
        systemData = data_import(Scenarios)

        
        # Step 1: Run the GT model with initial conditions
        #initial_run_GT(year)
        model = cooperative_game(year, scenario, systemData)
            
        
        # # Step 2: Run the OPGF with the initial conditions
        multinet = initial_run_OPGF(model,  systemData)
        


        genCapacities = model.q.extract_values()
        genCapacities = pd.DataFrame.from_dict(genCapacities, orient = 'index')
        # print(genCapacities)
        
        # cost_check(model, multinet, systemData)
        
        # Step 5: Goto step 2
        
        #############################
        
        from agg_results_handler import update_gt_with_opgf_results
        
        model = update_gt_with_opgf_results(model, multinet, systemData, selected_players)

        #############################
        
        Case_name = 'Cooperative'  
        
        if phi_fg==0.0:
            phi_nm = 'Variable'
        else: phi_nm = phi_fg
                       
                       
        Case_Demand = {k: Scenarios[k] for k in ['time_steps', "Demand_level"]}
                
        Subfolders = {'': Case_name} | {'phi': f"{phi_nm}"} | {'invstyrs': f"{invest_span}"} | {'price': f"{price_level}"} | {'Case': f"{gens_fg}"} | Case_Demand
        
        output_folder = "Output"
        for key, value in Subfolders.items():
            folder_name = f"{key}_{value}" if key else value
            output_folder = os.path.join(output_folder, folder_name)
            os.makedirs(output_folder, exist_ok=True)
        
       
        cost_results = cost_check(model, multinet, systemData)
        
        
        save_generation_results(model, selected_players, multinet, systemData, output_folder)
        
        save_simulation_results(model, genCapacities, multinet, cost_results, output_folder)
        
        save_cost_calc(systemData, output_folder)
        
        print_simulation_results(systemData, genCapacities, multinet, cost_results, output_folder)
        
        from agg_results_handler import process_policy_support_and_save
       
        process_policy_support_and_save(model, multinet, systemData, df_shap, output_folder,
                                     pos_shap_elec, pos_shap_hydrogen,neg_shap_elec, neg_shap_hydrogen,
                                     calculate_policy_opex_support, calculate_policy_npv_support)
    
    
    
        run_scenario_simulation(phi_fg, invest_span, gens_flag, time_steps, Scenarios['Demand_level'], output_folder)


        ##End time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('elapsed_time', elapsed_time)