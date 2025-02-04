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
from multinet_GB import *




def data_import(Scenarios):
    path = r"Refined FES scenario inputs/"
    GBdata_path = "GB_2050_Data/"  # Path to the Data folder

    systemData = {
        # 'Carbon Price': pd.read_excel(path + 'Carbon price/carbon_price.xlsx'),
        # 'Installed Capacity': pd.read_excel(path + 'InstalledCapacity/installed_capacity.xlsx', index_col=0),
        # 'Power Demand': pd.read_excel(path + 'PowerDemand_Heating_Input/powerDemand.xlsx', index_col=0),
        
        'Players': pd.read_csv(GBdata_path + 'GB_market_players.csv'), #import the parameters
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
        
   
    systemData['Power Demand'] = systemData['Bus Data']['2'],
    # systemData['Gas Consumption'] = pd.read_excel(path + 'gasConsumption_heating_input/gasCons_'+ scenario + '.xlsx', sheet_name = year, index_col = 0),
    
    
    # systemData['Players']['max_p_mw'] = systemData['Installed capacity'][2025]
    systemData['Installed capacity'][2050] *= 10000

    
    systemData['Scenarios_id'] =Scenarios
    
    # Filter out players with zero capacity (2050)
    systemData['players_2050'] = systemData['Players'][systemData['Players']["max_p_mw"] > 0]
    
    return systemData



def variable_list(model, year, data):
    ub = data['Installed capacity'][int(year)] - data['Installed capacity'][2025]
    def fb(model,i):
        return (None,ub[i])
    
    # Variable list -----------------------------------------------------------
    # Define the list of selected players
    
    global selected_players 
    selected_players = [6, 37, 43, 69, 87, 112, 132, 160, 174, 195, 214, 238, 258, 279, 307, 322, 342]
    # selected_players = list(range(0, 140))
    # selected_players = list(range(0, 357))
   
    # # Extract the IDs of the 2050 players 
    # selected_players =systemData['players_2050']["id"].tolist() ## 217 players

    
    #Initialize the model's set with the selected players
    model.p = pyoen.Set(initialize=selected_players)
    
    # model.p     = pyoen.Set(initialize=data['Players']['id'])
   
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
    model.Q_e = 22729.35
    
    # print(np.sum(data['Power Demand'])[0]) # Check the time period
    # # model.Q_e = pyoen.Param(initialize= 500) # value in MWh
    
    # model.Q_g = pyoen.Param(initialize=  max(np.sum(data['Gas Consumption']))*(39.41*3600)/(0.4*1000)) # value in MWh
    # # model.Q_g = pyoen.Param(initialize= max(np.sum(data['Gas Consumption'], axis=1))) # value in MWh
    # print(max(np.sum(data['Gas Consumption']))*(39.41*3600)/(0.4*1000*1000))
    # model.Q_h = pyoen.Param(initialize=(2))

    model.pE = pyoen.Param(initialize=(250)) # in £/MWh
    model.pG = pyoen.Param(initialize=(250))
    model.pH = pyoen.Param(initialize=(250))

    model.CO2 = pyoen.Param(initialize=(43.29)) #£/tonne C02
    

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
    # Define generation type categories
    renewable_types = ['Onshore Wind', 'Offshore Wind', 'PV', 'CSP', 'Biomass', 'Hydro ROR', 'Hydro reservoir', 'Geothermal', 'Other RES']
    fossil_fuel_types = ['Coal conventional', 'Gas Conventional', 'Coal CCS', 'Gas CCS', 'Peakers', 'OCGT', 'Gas(LF)', 'G2P']
    chp_types = ['Micro CHP', 'Industrial CHP']

    # Restrict summation to selected players only
    model.c5 = pyoen.Constraint(
        expr=(
            sum(model.q[i] for i in selected_players if data['Players']['type'][i] in renewable_types) +
            sum(model.q[i] for i in selected_players if data['Players']['type'][i] in fossil_fuel_types) +
            sum(model.q[i] for i in selected_players if data['Players']['type'][i] in chp_types) -
            model.Q_e == 0
        )
    )
    



    # model.c6 = pyoen.Constraint(expr= sum(model.q[i] for i in data['Players'].index[data['Players']['type']=='meth'].tolist()) +  
                                #sum(model.q[i] for i in data['Players'].index[data['Players']['type']=='P2G'].tolist()) - 
                                #sum(model.q[i] for i in data['Players'].index[data['Players']['type']=='CHP'].tolist()) - 
                                #sum(model.q[i] for i in data['Players'].index[data['Players']['type']=='GT'].tolist()) - 
                                # -model.Q_g == 0)
    
    # model.c7 = pyoen.Constraint(expr= model.q['EZ'] - model.q['FC']  == 0 )
    
    return model
    
    

def solver(model):
        
    # Solver options --------------------------------------------------------------
    opt = SolverFactory('pathampl', executable='pathampl.exe')
    # opt = SolverFactory('ipopt', executable='ipopt.exe')
    opt.options['max_iter'] = 10
    opt.solve(model, tee=False)
    model.display()



def cost_calc(model):
    
    total_cost = (sum(pyoen.value(model.q[tech] * model.c[tech]) for tech in model.p) + 
                  sum(pyoen.value(model.q[tech] * model.CO2*model.emis[tech]) for tech in model.p))
    
    return total_cost

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
    genData['8']  = genCapacities
    
    hour_of_day = 1 # Input how many hours the model shall run
    multinet = run_OPGF(hour_of_day, genData,  systemData) # Running the OPGF
    return multinet

def cost_check(model, multinet, systemData):
    cost_GT = cost_calc(model) # Importing the total cost from the GT model
    print('Total cost GT: ', cost_GT)
    
    players = systemData['Players']
    
    emissionCost = (sum(multinet['nets']['power']['res_gen']['p_mw'] * 43.29 * players['emissions'][selected_players]) +
                    sum(multinet.nets['hydrogen'].res_sink['mdot_kg_per_s'])*(39.41*3600)/(0.4*1000) * 43.29 * players['emissions'][29] +
                    sum(multinet['nets']['power']['res_load']['p_mw'][33:37]) * 43.29 * players['emissions'][25] 
                    )
    
    cost_OPGF = (sum(multinet['nets']['power']['res_gen']['p_mw'] * multinet['nets']['power']['poly_cost']['cp0_eur'])+
          sum(multinet.nets['hydrogen'].res_sink['mdot_kg_per_s'])*(39.41*3600)/(0.4*1000)*76 +
           sum(multinet['nets']['power']['res_load']['p_mw'][33:37]) * players['costs'][29] +
            + emissionCost)
    
    print('Total cost OPGF: ', cost_OPGF)
    
    print('cost difference: ', cost_GT - cost_OPGF )
    
def update_GT(year, scenario, multinet, systemData):
    systemData['Players']['max_p_mw'][selected_players] = multinet['nets']['power']['res_gen']['p_mw'][selected_players]    #do the change here
    systemData['Players']['max_p_mw'][357] = abs(sum(multinet.nets['hydrogen'].res_ext_grid['mdot_kg_per_s'])*(39.41*3600)/(0.4*1000))
    model = game(year, scenario, systemData)
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

    # scenario_options = [H2 option, Zone option, and Profile option]
    scenario_options = [1, 21, 5]  
    scenario_options = [2, 21, 5] 
    scenario_options = [3, 21, 5] 
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
    
    # Step 4: Change the GT value with new value and re-run the GT model
    model = update_GT(year, scenario, multinet, systemData)
    genCapacities = model.q.extract_values()
    genCapacities = pd.DataFrame.from_dict(genCapacities, orient = 'index')
    # cost_check(model, multinet, systemData)
    

    # Step 5: Goto step 2
    
    
    #############################
    
    
    print(f"Total Generation GTM: {genCapacities[0].sum()}")
    print(f"Total Generation OPF:{sum(multinet['nets']['power']['res_gen']['p_mw'])}")
    print(f"Total Demand: {model.Q_e}")

    cost_check(model, multinet, systemData)
    
    
    
    
    