import sys
import os
import pypsa 
import matplotlib.pyplot as plt
plt.style.use("bmh")
import pandas as pd
from pypsa.plot import add_legend_patches
import gurobipy
import cartopy.crs as ccrs
from pypsa.optimization import optimize
import matplotlib.cm as cm
import numpy as np
import xarray as xr
import seaborn as sns

def no_inv(n2,n):
    #set the optimal capacity of generators from the base scenario as the new minimum capacity 
    for index, value in n2.generators.p_nom_extendable.items():
        if value:  
            n2.generators.at[index, 'p_nom'] = n.generators.at[index, 'p_nom_opt']
            #n2.generators.at[index, 'p_nom_max'] = n.generators.at[index, 'p_nom_opt']
            n2.generators.at[index,'p_nom_extendable'] = False

    #set the optimal capacity of storage units from the base scenario as the new minimum capacity 
    for index, value in n2.storage_units.p_nom_extendable.items():
        if value:  
            n2.storage_units.at[index, 'p_nom'] = n.storage_units.at[index, 'p_nom_opt']
            #n2.storage_units.at[index, 'p_nom_max'] = n.storage_units.at[index, 'p_nom_opt']
            n2.storage_units.at[index,'p_nom_extendable'] = False

    #set the optimal capacity of stores from the base scenario as the new minimum capacity 
    for index, value in n2.stores.e_nom_extendable.items():
        if value:  
            n2.stores.at[index, 'e_nom'] = n.stores.at[index, 'e_nom_opt']
            #n2.stores.at[index, 'e_nom_max'] = n.stores.at[index, 'e_nom_opt']
            n2.stores.at[index, 'e_nom_extendable'] =False

    #set the optimal capacity of lines from the base scenario as the new minimum capacity 
    for index, value in n2.lines.s_nom_extendable.items():
        if value:  
            n2.lines.at[index, 's_nom'] = n.lines.at[index, 's_nom_opt']
            #n2.lines.at[index, 's_nom_max'] = n.lines.at[index, 's_nom_opt']
            n2.lines.at[index, 's_nom_extendable'] =False

    #set the optimal capacity of lines from the base scenario as the new minimum capacity 
    for index, value in n2.links.p_nom_extendable.items():
        if value:  
            n2.links.at[index, 'p_nom'] = n.links.at[index, 'p_nom_opt']
            #n2.lines.at[index, 's_nom_max'] = n.lines.at[index, 's_nom_opt']
            n2.links.at[index, 'p_nom_extendable'] =False

def solve_base(input_file, output_file, output_RH, co2_price, horizon):
    n = pypsa.Network(input_file)

    for index in n.generators.index:
        emission = n.carriers.co2_emissions.filter(like=n.generators.carrier[index]) # t/MWh_th
        n.generators.at[index, 'marginal_cost'] = n.generators.at[index, 'marginal_cost'] + (emission * co2_price / n.generators.at[index, 'efficiency']) # €/MWhel

    # Remove nuclear fleet from Germany
    if any(n.generators.index.str.startswith('DE') & n.generators.index.str.endswith('nuclear')):
        n.generators.loc[n.generators.index.str.startswith('DE') & n.generators.index.str.endswith('nuclear'), 'p_nom'] = 0

    # Set marginal cost of load shedding to 3000€/MWh
    for bus in n.generators.bus:
        for index in n.generators.index:
            if not bus.endswith('H2') and index.endswith('load'):
                n.generators.loc[index, 'marginal_cost'] = 3000

    # Edit load shedding's unit from kW to MW
    for index in n.generators.index:
        if index.endswith('load') and not index.endswith('H2 load'):
            n.generators.loc[index, 'sign'] = 1

    n.optimize(solver_name='gurobi')

    #BAse RH
    n_RH = n.copy()
    no_inv(n_RH,n)
    n_RH.storage_units['cylic_state_of_charge'] = False
    n_RH.storage_units['cylic_state_of_charge_per_period'] = False
    optimize.optimize_with_rolling_horizon(n_RH, horizon=int(horizon), overlap=0, solver_name='gurobi')

    # Save the solved network to the output file
    n.export_to_netcdf(output_file)
    n_RH.export_to_netcdf(output_RH)

    # Also save the solved network to the resources directory
    resource_output_file = os.path.join("resources", os.path.basename(output_file))
    n.export_to_netcdf(resource_output_file)

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    co2_price = float(sys.argv[3])
    output_RH = sys.argv[4]
    horizon = sys.argv[5]

    solve_base(input_file, output_file, output_RH, co2_price, horizon)

