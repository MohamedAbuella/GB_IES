# -*- coding: utf-8 -*-
"""
Created on Tu April 8 11:42:51 2025

@author: Mhdella
"""

import numpy as np
import pandas as pd
import os
import time
import pickle
from tqdm import tqdm
from pandapower.optimal_powerflow import OPFNotConverged  
import traceback

import pyomo.environ as pyoen
from pyomo.environ import *
# import pyomo.mpec as pyompec
from pyomo.opt import SolverFactory

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



def variable_list(model, year, systemData):
    
    global ub
    ub = systemData['Installed capacity'][int(year)] - systemData['Installed capacity'][2025]


    # Variable list -----------------------------------------------------------    
    global selected_players

    selected_players = get_selected_players('type') # Type of players
    
    # selected_players = get_selected_players('ele_players') # Only the elec  players
    
    
    q_init = {i: abs(systemData['Agg_Players']['max_p_mw'])[i] for i in selected_players}


    model.p = pyoen.Set(initialize=selected_players)
    
    if systemData['max_p_mw_updated']=='Un-Updated':
        model.q = Var(model.p, within=NonNegativeReals, bounds=(0, 300000), initialize=0)

    elif systemData['max_p_mw_updated'] == 'Updated':
        model.q = pyoen.Var(model.p, within=pyoen.NonNegativeReals, bounds=(0, 300000), initialize=q_init)

    model.I = Var(model.p, within=NonNegativeReals, initialize=0)
    model.sigma = Var(model.p, within=NonNegativeReals, initialize=0)
    model.delta = Var(model.p, within=NonNegativeReals, initialize=0)
    model.alpha = Var(model.p, initialize=0)
    model.lambda_eq = pyoen.Var()
        
    model.epsilon = pyoen.Var(bounds=(0, 1e12))

    # model.epsilon = pyoen.Var(bounds=(0, 2))


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
    
    # filtered_k = {i: data['Installed capacity'][2025][i] for i in model.p if i in data['Installed capacity'][2025]}
    # filtered_k = {i: abs(data['Agg_Players']['max_p_mw'])[i] for i in model.p if i in data['Agg_Players']['max_p_mw']}
    filtered_k = {i: data['Installed capacity'][2050][i] for i in model.p if i in data['Installed capacity'][2050]}

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


    model.lamda = pyoen.Param(model.p, initialize=filtered_lamda)  
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





def NLP_conditions(model):

    # Capacity constraint
    def capacity_constraint(model, i):
        return model.q[i] <= model.k[i] + model.I[i]

    model.CapacityConstraint = Constraint(
        model.p, rule=capacity_constraint
    )

    # Investment upper bound
    def investment_constraint(model, i):
        return model.I[i] <= model.iMAX[i]

    model.InvestmentConstraint = Constraint(
        model.p, rule=investment_constraint
    )

    return model



def system_cost_objective(model):
    return sum(
        (
            model.c[i] * model.q[i]                 # variable generation cost
            + model.CO2 * model.emis[i] * model.q[i]  # emissions cost
        ) / model.AF[i]
        + model.phi[i] * model.eps[i] * model.I[i]   # annualized investment cost
        for i in model.p
    )




def energy_balance_rule(model):
    return sum(model.q[i] for i in model.p) >= model.Q_e + model.Q_h 



def solver(model):
    
    # # # Set the PATH solver license dynamically in Python
    # os.environ["PATH_LICENSE_STRING"] = "2830898829&Courtesy&&&USR&45321&5_1_2021&1000&PATH&GEN&31_12_2025&0_0_0&6000&0_0"
        
    # Solver options --------------------------------------------------------------
    # opt = SolverFactory('pathampl', executable='pathampl.exe')
    # opt = SolverFactory('ipopt', executable='ipopt.exe')
    
    # # Use the pathampl.exe from the current directory
    # opt = SolverFactory('pathampl', executable=os.path.join(os.getcwd(), 'pathampl.exe'))
    
    
    ## Use the pathampl.exe from the current directory
    opt = SolverFactory('ipopt', executable=os.path.join(os.getcwd(), 'ipopt.exe'))
    # opt = SolverFactory('ipopt')
    
    opt.options['max_iter'] = 500
    opt.options['tol'] = 1e-6
    opt.options['constr_viol_tol'] = 1e-6
    


    # opt.options['feasibility_tol'] = 1e-2
    # opt.options['constr_viol_tol'] = 1e-2
    # opt.options['allow_infeasibilities'] = 'no'  # Strict mode
    # opt.options['reset'] = 'yes'
    
    # opt.options['allow_infeasibilities'] = 'yes'

    
    opt.solve(model, tee=False)
    
    
    # model.display()
    
    return model




def game(year, scenario, systemData):
    
    model = pyoen.ConcreteModel()

    variable_list(model, year, systemData)   
 
    parameter_list(model, year, systemData)
    
    # complementarity_conditions(model)
    
    # equality_constraints(model, systemData)
    
        
    
    model.SystemProfit = Objective(rule=system_cost_objective, sense=minimize)


    NLP_conditions(model)

    model.EnergyBalance = Constraint(rule=energy_balance_rule)
    
    
    solver(model)
        

    # for i in selected_players:
    #     print(f"Player: {i} -> q[i]: {pyoen.value(model.q[i])}, k[i]: {pyoen.value(model.k[i])},  iMAX[i]: {pyoen.value(model.iMAX[i])}, I[i]: {pyoen.value(model.I[i])}, delta[i]: {pyoen.value(model.delta[i])}")
    

    return model




def initial_run_OPGF(model, systemData):
    
    genCapacities = model.q.extract_values()
    genCapacities = pd.DataFrame.from_dict(genCapacities, orient='index')
    
    updated_gen_data, updated_p2g_data, updated_g2g_data, updated_g2p_data = distribute_players_capacities(genCapacities, systemData)
    
    updated_gen_data['out_pmw'] = updated_gen_data['out_pmw'] * systemData['Players'].scale
    
    
    systemData['Gen Data']['8'] = updated_gen_data['out_pmw']
    

    storage_mask = updated_gen_data['type'] == 'Storage'
    systemData['Gen Data'].loc[storage_mask, '9'] = -abs(updated_gen_data.loc[storage_mask, 'out_pmw'])
    
    
    systemData['P2G Data'] = updated_p2g_data
    
    systemData['G2G Data'] = updated_g2g_data
    
    systemData['G2P Data'] = updated_g2p_data


    # Pick an hour for the model to run
    hour_of_day = time_steps
    
    multinet = run_OPGF(hour_of_day, systemData['Gen Data'], systemData)  # Running the OPGF
    
    return multinet





def cost_check(model, multinet, systemData):
    
    players = systemData['Players']
    
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
    
    # op_c_g2g = c_g2g + 0
    op_c_g2g = c_g2g + 80  # Cost of Gas fuel + CCS
    
    g2gCost= g2g_in * op_c_g2g + CO2_cost_G2G

    cost_GT = cost_GT + co2_cost_GT + g2gCost
    cost_OPGF = cost_OPGF + co2_cost_OPGF + g2gCost 
    
    # # Normalized cost difference for convergence
    # cost_diff = abs(cost_GT - cost_OPGF) / max(abs(cost_GT), 1e-6)  # Avoid division by zero
    
    cost_diff = cost_GT - cost_OPGF
    
    return {
        "cost_GT": cost_GT,
        "cost_OPGF": cost_OPGF,
        "cost_diff": cost_diff,
        "emission_GT": emission_GT,
        "co2_cost_GT": co2_cost_GT,
        "emission_OPGF": emission_OPGF,
        "co2_cost_OPGF": co2_cost_OPGF,
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
    
    
    
    model = game(year, scenario, systemData)
    
        
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
    players_ids = systemData['Agg_Players'].loc[systemData['Players']['type'].isin(selected_types)].drop_duplicates(subset=['type'])[['type', 'id']]
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
    
    # # Set 'max_p_mw' to 0 for players not in selected list
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





# def run_iterative_bilevel(year, scenario, systemData, max_iter=100, epsilon_qty=0.01, epsilon_cost=0.01):
def run_iterative_bilevel(year, scenario, systemData, max_iter, epsilon_qty, epsilon_cost):
    
    # Initialize a list to store convergence data
    global convergence_data
    convergence_data = []

    # Initialize the model and run the first iteration
    try:
        model = game(year, scenario, systemData)
        multinet = initial_run_OPGF(model, systemData)
        cost_results = cost_check(model, multinet, systemData)

        # Log initial run as iteration=1
        genCapacities = pd.DataFrame.from_dict(model.q.extract_values(), orient='index')
        selected_players = systemData.get('selected_players', [])
        total_gt_gens = genCapacities.loc[genCapacities.index.intersection(selected_players)].sum().sum() if selected_players else float('nan')
        total_opgf_gens = abs(systemData['Agg_Players']['max_p_mw'].loc[selected_players]).sum() if selected_players else float('nan')
        qty_diff = np.abs(total_gt_gens - total_opgf_gens)
        cost_diff = cost_results['cost_diff']

        convergence_data.append({
            'iteration': 1,
            'gen_diff': qty_diff,
            'cost_diff': cost_diff,
            'gen_GTM': total_gt_gens,
            'gen_OPGF': total_opgf_gens,
            'cost_GTM': cost_results['cost_GT'],
            'cost_OPGF': cost_results['cost_OPGF'],
            'status': 'success'
        })

    except OPFNotConverged as e:
        print(f"OPGF failed to converge in initial iteration: {str(e)}")
        output_folder = systemData.get('output_folder', 'Output')
        genCapacities = pd.DataFrame.from_dict(model.q.extract_values(), orient='index')
        convergence_data.append({
            'iteration': 1,
            'gen_diff': float('nan'),
            'cost_diff': float('nan'),
            'gen_GTM': float('nan'),
            'gen_OPGF': float('nan'),
            'cost_GTM': float('nan'),
            'cost_OPGF': float('nan'),
            'status': 'OPGF_failed'
        })
        pd.DataFrame(convergence_data).to_csv(os.path.join(output_folder, 'convergence.csv'), index=False)
        print("Initial quantities:", genCapacities.to_dict())
        print("Initial max_p_mw:", systemData['Agg_Players']['max_p_mw'].to_dict())
        print("Initial Q_e:", systemData.get('Q_e', 'Not found'), "Q_h:", systemData.get('Q_h', 'Not found'))
        return model, None, None
    except Exception as e:
        print(f"Unexpected error in initial iteration: {str(e)}")
        print("Traceback:", traceback.format_exc())
        output_folder = systemData.get('output_folder', 'Output')
        convergence_data.append({
            'iteration': 1,
            'gen_diff': float('nan'),
            'cost_diff': float('nan'),
            'gen_GTM': float('nan'),
            'gen_OPGF': float('nan'),
            'cost_GTM': float('nan'),
            'cost_OPGF': float('nan'),
            'status': 'unexpected_error'
        })
        pd.DataFrame(convergence_data).to_csv(os.path.join(output_folder, 'convergence.csv'), index=False)
        return model, None, None

    # Track previous differences for stabilization check
    prev_qty_diff = float('inf')
    prev_cost_diff = float('inf')
    prev_max_p_mw = systemData['Agg_Players']['max_p_mw'].copy()

    # Validate selected_players against max_p_mw indices
    selected_players = systemData.get('selected_players', [])
    max_p_mw = systemData['Agg_Players']['max_p_mw']
    max_p_mw_indices = max_p_mw.index if isinstance(max_p_mw, (pd.Series, pd.DataFrame)) else list(max_p_mw.keys())
    if not all(p in max_p_mw_indices for p in selected_players):
        print(f"Error: Some selected_players {selected_players} not in max_p_mw indices: {max_p_mw_indices}")
        convergence_data.append({
            'iteration': 1,
            'gen_diff': float('nan'),
            'cost_diff': float('nan'),
            'gen_GTM': float('nan'),
            'gen_OPGF': float('nan'),
            'cost_GTM': float('nan'),
            'cost_OPGF': float('nan'),
            'status': 'max_p_mw_index_error'
        })
        output_folder = systemData.get('output_folder', 'Output')
        pd.DataFrame(convergence_data).to_csv(os.path.join(output_folder, 'convergence.csv'), index=False)
        return model, None, None

    # Iteration loop starting at iteration=1 (logged as iteration=2)
    try:
        with tqdm(total=max_iter, desc="Iterative Bi-Level Optimization", unit="iter",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                  colour='green') as pbar:
            for iteration in range(max_iter):
                try:
                    prev_genCapacities = pd.DataFrame.from_dict(model.q.extract_values(), orient='index')
                    prev_cost_GT = cost_results['cost_GT']
                    prev_cost_OPGF = cost_results['cost_OPGF']
                except Exception as e:
                    print(f"Error storing previous quantities at iteration {iteration+2}: {str(e)}")
                    print("Traceback:", traceback.format_exc())
                    convergence_data.append({
                        'iteration': iteration + 2,
                        'gen_diff': float('nan'),
                        'cost_diff': float('nan'),
                        'gen_GTM': float('nan'),
                        'gen_OPGF': float('nan'),
                        'cost_GTM': float('nan'),
                        'cost_OPGF': float('nan'),
                        'status': 'prev_quantities_error'
                    })
                    output_folder = systemData.get('output_folder', 'Output')
                    pd.DataFrame(convergence_data).to_csv(os.path.join(output_folder, 'convergence.csv'), index=False)
                    pbar.close()
                    return model, multinet, cost_results

                try:
                    # Update GT with OPGF feedback
                    model = update_GT(year, scenario, multinet, systemData)
                    multinet = initial_run_OPGF(model, systemData)
                    cost_results = cost_check(model, multinet, systemData)

                    # Check quantity convergence
                    genCapacities = pd.DataFrame.from_dict(model.q.extract_values(), orient='index')
                    total_gt_gens = genCapacities.loc[genCapacities.index.intersection(selected_players)].sum().sum() if selected_players else float('nan')
                    total_opgf_gens = abs(systemData['Agg_Players']['max_p_mw'].loc[selected_players]).sum() if selected_players else float('nan')
                    qty_diff = np.abs(total_gt_gens - total_opgf_gens)
                    cost_diff = cost_results['cost_diff']

                    # Log convergence data
                    convergence_data.append({
                        'iteration': iteration + 2,
                        'gen_diff': qty_diff,
                        'cost_diff': cost_diff,
                        'gen_GTM': total_gt_gens,
                        'gen_OPGF': total_opgf_gens,
                        'cost_GTM': cost_results['cost_GT'],
                        'cost_OPGF': cost_results['cost_OPGF'],
                        'status': 'success'
                    })

                    pbar.set_postfix({
                        'gen_diff': f'{qty_diff:.4f}',
                        'cost_diff': f'{cost_diff:.4f}',
                        'gen_GTM': f'{total_gt_gens:.2f}',
                        'gen_OPGF': f'{total_opgf_gens:.2f}',
                        'cost_GTM': f'{cost_results["cost_GT"]:.2f}',
                        'cost_OPGF': f'{cost_results["cost_OPGF"]:.2f}',
                    })
                    pbar.update(1)

                    if qty_diff < epsilon_qty and cost_diff < epsilon_cost:
                        print(f"Converged after {iteration+2} iterations: qty_diff={qty_diff:.4e}, cost_diff={cost_diff:.4e}")
                        pbar.close()
                        break

                    if abs(qty_diff - prev_qty_diff) < 1e-6 and abs(cost_diff - prev_cost_diff) < 1e-6 and iteration > 0:
                        print(f"Stopped after {iteration+2} iterations: Differences stabilized (qty_diff={qty_diff:.4e}, cost_diff={cost_diff:.4e})")
                        pbar.close()
                        break

                    prev_qty_diff = qty_diff
                    prev_cost_diff = cost_diff
                    prev_max_p_mw = systemData['Agg_Players']['max_p_mw'].copy()

                except OPFNotConverged as e:
                    print(f"OPGF failed to converge at iteration {iteration+2}: {str(e)}")
                    systemData['Agg_Players']['max_p_mw'] = prev_max_p_mw.copy()
                    convergence_data.append({
                        'iteration': iteration + 2,
                        'gen_diff': float('nan'),
                        'cost_diff': float('nan'),
                        'gen_GTM': float('nan'),
                        'gen_OPGF': float('nan'),
                        'cost_GTM': float('nan'),
                        'cost_OPGF': float('nan'),
                        'status': 'OPGF_failed'
                    })
                    output_folder = systemData.get('output_folder', 'Output')
                    pd.DataFrame(convergence_data).to_csv(os.path.join(output_folder, 'convergence.csv'), index=False)
                    print("Failed quantities:", prev_genCapacities.to_dict())
                    print("Failed max_p_mw:", systemData['Agg_Players']['max_p_mw'].to_dict())
                    print("Failed Q_e:", systemData.get('Q_e', 'Not found'), "Q_h:", systemData.get('Q_h', 'Not found'))
                    pbar.close()
                    return model, multinet, cost_results
                except Exception as e:
                    print(f"Unexpected error at iteration {iteration+2}: {str(e)}")
                    print("Traceback:", traceback.format_exc())
                    convergence_data.append({
                        'iteration': iteration + 2,
                        'gen_diff': float('nan'),
                        'cost_diff': float('nan'),
                        'gen_GTM': float('nan'),
                        'gen_OPGF': float('nan'),
                        'cost_GTM': float('nan'),
                        'cost_OPGF': float('nan'),
                        'status': 'unexpected_error'
                    })
                    output_folder = systemData.get('output_folder', 'Output')
                    pd.DataFrame(convergence_data).to_csv(os.path.join(output_folder, 'convergence.csv'), index=False)
                    print("Last quantities:", prev_genCapacities.to_dict())
                    print("Last max_p_mw:", systemData['Agg_Players']['max_p_mw'].to_dict())
                    print("Last Q_e:", systemData.get('Q_e', 'Not found'), "Q_h:", systemData.get('Q_h', 'Not found'))
                    pbar.close()
                    return model, multinet, cost_results

                if iteration == max_iter - 1:
                    print("Max iterations reached without convergence")
                    pbar.close()

            output_folder = systemData.get('output_folder', 'Output')
            pd.DataFrame(convergence_data).to_csv(os.path.join(output_folder, 'convergence.csv'), index=False)
            # print("Final quantities:", genCapacities.to_dict())
            # print("Final max_p_mw:", systemData['Agg_Players']['max_p_mw'].to_dict())
            # print("Final Q_e:", systemData.get('Q_e', 'Not found'), "Q_h:", systemData.get('Q_h', 'Not found'))

    except KeyboardInterrupt:
        print("Iteration interrupted by user")
        output_folder = systemData.get('output_folder', 'Output')
        pd.DataFrame(convergence_data).to_csv(os.path.join(output_folder, 'convergence.csv'), index=False)
        print("Interrupted quantities:", genCapacities.to_dict())
        print("Interrupted max_p_mw:", systemData['Agg_Players']['max_p_mw'].to_dict())
        print("Interrupted Q_e:", systemData.get('Q_e', 'Not found'), "Q_h:", systemData.get('Q_h', 'Not found'))
        return model, multinet, cost_results

    return model, multinet, cost_results






def choose_scenario_from_list(options_dict, choice):

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
    global phi_fg, gens_fg
    
    
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
        
        
        #############################
        
        Case_name = 'Benchmark'  
        
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
            
            
        systemData['output_folder'] = output_folder
    
        # # Run iterative bi-level optimization
        model, multinet, cost_results = run_iterative_bilevel(year, scenario, systemData, max_iter=10, epsilon_qty=0.1, epsilon_cost=0.1)
    
    
        # model = game(year, scenario, systemData)
        # multinet = initial_run_OPGF(model, systemData)
        
        # model = update_GT(year, scenario, multinet, systemData)
        # multinet = initial_run_OPGF(model,  systemData)
        
        # model = update_GT(year, scenario, multinet, systemData)
        # multinet = initial_run_OPGF(model,  systemData)
    
    
        cost_results = cost_check(model, multinet, systemData)
        
    
        genCapacities = pd.DataFrame.from_dict(model.q.extract_values(), orient='index')
        
        save_generation_results(model, selected_players, multinet, systemData, output_folder)
        
        save_simulation_results(model, genCapacities, multinet, cost_results, output_folder)
        
        save_cost_calc(systemData, output_folder)
        
        print_simulation_results(systemData, genCapacities, multinet, cost_results, output_folder)
    
        plot_convergence(output_folder)
        
        # # convergence_data
        
        
        # # Save systemData as pickle
        # pickle_path = os.path.join("Output", Case_name + '_systemData.pkl')
        # with open(pickle_path, 'wb') as f:
        #     pickle.dump(systemData, f)
            
        
    
        run_nash_scenario_simulation(phi_fg, gens_flag, time_steps, Scenarios['Demand_level'], output_folder)

        ##End time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('elapsed_time', elapsed_time)