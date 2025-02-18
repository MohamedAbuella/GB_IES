# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 06:05:41 2025

@author: Mhdella
"""

import numpy as np
import pandas as pd
import os
import pyomo.environ as pyoen
from pyomo.environ import *
import pyomo.mpec as pyompec
from pyomo.opt import SolverFactory
from results_handler import *
from multinet_GB import *


import warnings
warnings.filterwarnings('ignore')  # Ignores all warnings

def data_import(Scenarios):
    path = r"Refined FES scenario inputs/"
    GBdata_path = "GB_2050_Data/"  # Path to the Data folder

    Players_data = pd.read_csv(GBdata_path + 'GB_market_players.csv')
    Economics_data = pd.read_excel(GBdata_path + 'GB_Economic_Parameters.xlsx', sheet_name='Econ_Players', index_col = 0)
    # Economics_data = pd.read_excel(GBdata_path + 'GB_Economic_Parameters2.xlsx', sheet_name='Econ_Players', index_col = 0)

    Players_economics = pd.concat([Economics_data] * 17, ignore_index=True)
    Players_economics[['id', 'zone_id', 'max_p_mw']] = Players_data[['id', 'zone_id', 'max_p_mw']]

    
    systemData = {

        'Players':  Players_economics,
        'Economics': Economics_data,
        'Economics_H2': pd.read_excel(GBdata_path + 'GB_Economic_Parameters.xlsx', sheet_name='Econ_H2', index_col = 0),

        'Carbon Price'      : pd.read_excel(GBdata_path + 'carbon_price.xlsx'), # carbon price ,
        'Installed capacity': pd.read_excel(GBdata_path + 'installed_capacity.xlsx', sheet_name=scenario, index_col = 0), #installed capacity
        
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
        
    }
    
    
    # Process P & Q (demand) for Bus data (Q is not needed, because DC power flow)
    systemData['Bus Data']['2']=systemData['Zone_Demand'][Scenarios['Zone_Demand']].values # P(demand)
    # systemData['Bus Data']['3']= 0.3 * systemData['Zone_Demand'][Scenarios['Zone_Demand']].values  # Q(demand)
        
    systemData['Gen Data']['8'] = systemData['Gen Data']['8']

    
    # Normalize the values by dividing each column by its sum
    systemData['GB Demand Profiles'] = systemData['GB Demand Profiles'].div(systemData['GB Demand Profiles'].sum(axis=0), axis=1)*1e6

                                                         
    
    # Process time series profile
    systemData['Time Series Profiles'] = {
        'e_load_time_series': systemData['GB Demand Profiles'][Scenarios['Profile']].values,
        
        'g_load_time_series': systemData['GB Demand Profiles']['H2 industry'].values,
                                         }
        
   
    systemData['Power Demand'] = systemData['Bus Data']['2'],
    # systemData['Gas Consumption'] = pd.read_excel(path + 'gasConsumption_heating_input/gasCons_'+ scenario + '.xlsx', sheet_name = year, index_col = 0),
    
    
    # systemData['Players']['max_p_mw'] = systemData['Installed capacity'][2025]
    systemData['Installed capacity'][2050] *= 1.2

    
    systemData['Scenarios_id'] =Scenarios
    
    # # Filter out players with zero capacity (2050)
    systemData['Players_2050'] = systemData['Players'][systemData['Players']["max_p_mw"] > 0]
    
    multinet_0 = intital_multinet(time_steps,  systemData)

    systemData['Q_e'] = sum(multinet_0.nets['power'].load['p_mw'])
    
    return systemData



def variable_list(model, year, data):
    ub = data['Installed capacity'][int(year)] - data['Installed capacity'][2025]
    def fb(model,i):
        return (None,ub[i])
    
    # Variable list -----------------------------------------------------------
    # Define the list of selected players
    
    global selected_players 
    

    # # Extract the IDs of the 2050 players 
    selected_players = systemData['Players_2050']["id"].tolist() ## 141 players

    
    # #Initialize the model's set with the selected players
    model.p = pyoen.Set(initialize=selected_players)
       
    model.q     = pyoen.Var(model.p)
    model.I     = pyoen.Var(model.p)
    model.sigma = pyoen.Var(model.p)
    model.delta = pyoen.Var(model.p)
    model.alpha = pyoen.Var(model.p)
    
    return model




def parameter_list(model, year, data):
    # Parameter list ----------------------------------------------------------
    model.years = pyoen.Param(initialize=int(year) - 2025)  # value in 'years'
    model.rate = pyoen.Param(initialize=0.05)  # value in percent

    model.AF = pyoen.Param(initialize=((1 + model.rate) ** model.years - 1)
                           / (model.rate * (1 + model.rate) ** model.years))  # Annuity factor

    # Filter the relevant parameters for selected players
    filtered_costs = {i: data['Players']['costs'][i] for i in model.p if i in data['Players']['costs']}
    filtered_epsilon = {i: data['Players']['epsilon'][i] for i in model.p if i in data['Players']['epsilon']}
    filtered_phi = {i: data['Players']['phi'][i] for i in model.p if i in data['Players']['phi']}
    filtered_lamda = {i: data['Players']['economic_life'][i] for i in model.p if i in data['Players']['economic_life']}
    filtered_k = {i: data['Players']['max_p_mw'][i] for i in model.p if i in data['Players']['max_p_mw']}
    filtered_emissions = {i: data['Players']['emissions'][i] for i in model.p if i in data['Players']['emissions']}
    filtered_installed_capacity = {
        i: data['Installed capacity'][int(year)][i] - data['Installed capacity'][2025][i]
        for i in model.p if i in data['Installed capacity'][int(year)]
    }

    # Initialize the model parameters
    model.c = pyoen.Param(model.p, initialize=filtered_costs)  # value in £/MWh
    model.eps = pyoen.Param(model.p, initialize=filtered_epsilon)  # value in £/MWh
    model.phi = pyoen.Param(model.p, initialize=filtered_phi)  # value in percent
    model.lamda = pyoen.Param(model.p, initialize=filtered_lamda)  # value in years
    model.k = pyoen.Param(model.p, initialize=filtered_k)  # value in MW
    model.emis = pyoen.Param(model.p, initialize=filtered_emissions)  # tonne CO2/ MWh
    model.iMAX = pyoen.Param(model.p, initialize=filtered_installed_capacity)  # in MW

    # print(data['Installed capacity'][int(year)]-data['Installed capacity'][2025])
     
    # # model.Q_e = pyoen.Param(initialize= max(np.sum(data['Power Demand'], axis=1))) # value in MWh
    # model.Q_e = pyoen.Param(initialize= np.max(data['Power Demand'])) 
    # model.Q_e = 22729.35
    # model.Q_e = 112617.51

    model.Q_e = systemData['Q_e'] 

    
    # print(np.sum(data['Power Demand'])[0]) # Check the time period
    # # model.Q_e = pyoen.Param(initialize= 500) # value in MWh
    
    # model.Q_g = pyoen.Param(initialize=  max(np.sum(data['Gas Consumption']))*(39.41*3600)/(0.4*1000)) # value in MWh
    # # model.Q_g = pyoen.Param(initialize= max(np.sum(data['Gas Consumption'], axis=1))) # value in MWh
    # print(max(np.sum(data['Gas Consumption']))*(39.41*3600)/(0.4*1000*1000))
    # model.Q_h = pyoen.Param(initialize=(2))
    
    model.pE = pyoen.Param(initialize=(33.69)) # in £/MWh
    model.pG = pyoen.Param(initialize=(9.73))
    model.pH = pyoen.Param(initialize=(9))

    # model.pE = pyoen.Param(initialize=(250)) # in £/MWh
    # model.pG = pyoen.Param(initialize=(250))
    # model.pH = pyoen.Param(initialize=(250))
    

    model.CO2 = pyoen.Param(initialize=(100)) #£/tonne C02
    

    return model



def complementarity_conditions(model):
    
    def technical_rule(model,i):
        return (pyompec.complements(model.q[i] >= 0, 
                                    model.AF * (model.pE - model.c[i]- model.CO2*model.emis[i]) - model.sigma[i]
                                    - model.alpha[i] <= 0))

    def investment_rule(model,i):
        return (pyompec.complements(model.I[i] >= 0, 
                                    (-model.AF * model.phi[i] * model.eps[i]) - model.eps[i] 
                                    * (1 - (model.lamda[i] - model.years)
                                       /(model.lamda[i] * (1 + model.rate)** model.years))
                                    + model.sigma[i] - model.delta[i] <= 0))
    
    def technical_limit(model,i):
        return(pyompec.complements(model.sigma[i] >= 0,
                                   model.q[i] - model.k[i] - model.I[i] <= 0))
    

    def investment_limit(model,i):
        return(pyompec.complements(model.delta[i] >= 0,
                                   model.I[i] - model.iMAX[i] <=0))
    
    model.c1 = pyompec.Complementarity(model.p, rule= technical_rule)
    model.c2 = pyompec.Complementarity(model.p, rule= investment_rule)
    model.c3 = pyompec.Complementarity(model.p, rule= technical_limit)
    model.c4 = pyompec.Complementarity(model.p, rule= investment_limit)
    
    return model



def equality_constraints(model, data):
    

    model.c5 = pyoen.Constraint(
        expr=sum(model.q[i] for i in selected_players) - model.Q_e == 0)



    # model.c6 = pyoen.Constraint(expr= sum(model.q[i] for i in data['Players'].index[data['Players']['type']=='meth'].tolist()) +  
                                #sum(model.q[i] for i in data['Players'].index[data['Players']['type']=='P2G'].tolist()) - 
                                #sum(model.q[i] for i in data['Players'].index[data['Players']['type']=='CHP'].tolist()) - 
                                #sum(model.q[i] for i in data['Players'].index[data['Players']['type']=='GT'].tolist()) - 
                                # -model.Q_g == 0)
    
    # model.c7 = pyoen.Constraint(expr= model.q['EZ'] - model.q['FC']  == 0 )
    
    return model
    
    

def solver(model):
    
    # # Set the PATH solver license dynamically in Python
    os.environ["PATH_LICENSE_STRING"] = "2830898829&Courtesy&&&USR&45321&5_1_2021&1000&PATH&GEN&31_12_2025&0_0_0&6000&0_0"
        
    # Solver options --------------------------------------------------------------
    opt = SolverFactory('pathampl', executable='pathampl.exe')
    # opt = SolverFactory('ipopt', executable='ipopt.exe')
    opt.options['max_iter'] = 10
    opt.solve(model, tee=False)
    # model.display()




def game(year, scenario, data):
    
    model = pyoen.ConcreteModel()
    
    variable_list(model, year, data)
    
    parameter_list(model, year, data)
    
    complementarity_conditions(model)
    
    equality_constraints(model, data)
    
    solver(model)
    
    # print_results(model)
    
    return model


def initial_run_OPGF(model, systemData):
    # Running the GT model
    genCapacities = model.q.extract_values()
    genCapacities = pd.DataFrame.from_dict(genCapacities, orient = 'index')
    
    # genData = pd.read_csv('Data/genData.csv')
    genData = systemData['Gen Data'].iloc[selected_players]
    genData['8'] = genCapacities
    systemData['Gen Data']['8'].iloc[selected_players] = genCapacities[0]
    
    hour_of_day = 1 # Input how many hours the model shall run
    multinet = run_OPGF(hour_of_day, genData,  systemData) # Running the OPGF
    return multinet



def cost_calc(model):
    return sum(pyoen.value(model.q[tech] * model.c[tech]) for tech in model.p)



def cost_check(model, multinet, systemData):
    cost_GT = cost_calc(model)
    players = systemData['Players']

    # CO2 emissions for GT Model
    emission_GT = sum(pyoen.value(model.q[tech] * model.emis[tech]) for tech in model.p)
    co2_cost_GT = emission_GT * pyoen.value(model.CO2)

    # CO2 emissions for OPGF Model
    emission_OPGF = sum(multinet['nets']['power']['res_gen']['p_mw'][selected_players] * players['emissions'][selected_players])
    co2_cost_OPGF = emission_OPGF * pyoen.value(model.CO2)

    # Emission cost calculation
    emissionCost = (emission_OPGF +
                    sum(multinet.nets['gas'].sink['mdot_kg_per_s']) * (14.64 * 3600) / 1000 * pyoen.value(model.CO2) * 0.320)

    cost_OPGF = (sum(multinet['nets']['power']['res_gen']['p_mw'][selected_players] * players['costs'][selected_players]) +
                 sum(multinet.nets['gas'].sink['mdot_kg_per_s']) * (14.64 * 3600) / 1000 * 85 +
                 (multinet.nets['hydrogen'].res_source['mdot_kg_per_s'].sum(skipna=True) * (39.41 * 3600) / 1000) * 110 +
                 emissionCost)
    
    return {
        "cost_GT": cost_GT,
        "cost_OPGF": cost_OPGF,
        "emission_GT": emission_GT,
        "co2_cost_GT": co2_cost_GT,
        "emission_OPGF": emission_OPGF,
        "co2_cost_OPGF": co2_cost_OPGF
           }



    
def update_GT(year, scenario, multinet, systemData):
    systemData['Players']['max_p_mw'][selected_players] = multinet['nets']['power']['res_gen']['p_mw'][selected_players] #do the change here
    # systemData['Players']['max_p_mw'][357] = abs(sum(multinet.nets['hydrogen'].res_ext_grid['mdot_kg_per_s'])*(39.41*3600)/(1000))

    model = game(year, scenario, systemData)
    
    # multinet['nets']['power']['gen']['p_mw'][selected_players] = pd.Series(model.q.extract_values())
    
    return model



    

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

    # print("\nSelected Scenarios:")
    # for key, value in Scenarios.items():
    #     print(f"{key}: {value}")
    
    

#################################################

    years = ['2025','2030','2035','2040','2045','2050']
    scenarios = ['FS', 'CT', 'LW', 'ST']
    
    year = years[5]
    scenario = scenarios[2]
    
#################################################

    time_steps = 0
    # time_steps = 1
    # time_steps = 2
    # time_steps = 12
    
    
    systemData = data_import(Scenarios)


    
    # Step 1: Run the GT model with initial conditions
    #initial_run_GT(year)
    model = game(year, scenario, systemData)
    
    
    # Step 2: Run the OPGF with the initial conditions
    multinet = initial_run_OPGF(model,  systemData)
    
    # Step 3: Compare the closeness of the results of OPGF model with the GT Model.
    # if the there is a mismatch then re-run the GT model with the latest OPGF value.
    # Calculate the cost in GT and OPGF scenarios. 
    # If the values are less than epsilon change then stop
    cost_check(model, multinet, systemData)
    
    # # Step 4: Change the GT value with new value and re-run the GT model
    model = update_GT(year, scenario, multinet, systemData)
    
    # multinet = initial_run_OPGF(model,  systemData)
    # model = update_GT(year, scenario, multinet, systemData)

    genCapacities = model.q.extract_values()
    genCapacities = pd.DataFrame.from_dict(genCapacities, orient = 'index')
    # cost_check(model, multinet, systemData)
    

    # Step 5: Goto step 2
    
    
    #############################
    
    cost_results = cost_check(model, multinet, systemData)
    
    save_generation_results(model, selected_players, multinet, systemData, output_folder="Output")
    
    save_simulation_results(model, genCapacities, multinet, cost_results, output_folder="Output")



    
    