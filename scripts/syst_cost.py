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

def syst_cost(n,country):
    #add column country in generaror, storage unit, lines and links
    for index, row in n.generators.iterrows():
        n.generators.at[index,'country'] = row['bus'][:2]

    for index, row in n.storage_units.iterrows():
        n.storage_units.at[index,'country'] = row['bus'][:2]

    for index, row in n.lines.iterrows():
        if country in row['bus0'] or country in row['bus1']:
            n.lines.at[index, 'country'] = country
        else:
            n.lines.at[index,'country'] = row['bus0'][:2]

    for index, row in n.links.iterrows():
        if country in row['bus0'] or country in row['bus1']:
            n.links.at[index, 'country'] = country
        else:
            n.links.at[index,'country'] = row['bus0'][:2]

    #calc opex and capex of the chosen country
    cost_df = pd.DataFrame(columns=['opex', 'capex'], index = n.carriers.index)

    opex = n.statistics.opex(comps=["Generator", "StorageUnit","Line","Link","Transformer"], groupby=["carrier","country"], aggregate_groups="sum").unstack()[country].fillna(0).droplevel(0)
    capex= n.statistics.capex(comps=["Generator", "StorageUnit","Line","Link","Transformer"], groupby=["carrier","country"], aggregate_groups="sum").unstack()[country].fillna(0).droplevel(0)

    for index, value in capex.items():
        cost_df.at[index, 'capex'] = value

    for index, value in opex.items():
        cost_df.at[index, 'opex'] = value

    cost_df = cost_df.fillna(0)
    system_cost = cost_df.sum(axis=1)/1e9 # Bill€

    stat = n.statistics()
    tsc = stat['Capital Expenditure'] + stat['Operational Expenditure'] # Bill €/a

    return system_cost, tsc.droplevel(0).div(1e9)

def main():
    input_file = sys.argv[1]
    country = sys.argv[2]
    output_country = sys.argv[3]
    output_tot = sys.argv[4]
    n = pypsa.Network(input_file)
    sys_cost_country, sys_cost_tot = syst_cost(n, country)
    sys_cost_country.to_csv(output_country)
    sys_cost_tot.to_csv(output_tot)

if __name__ == "__main__":
    # The script takes multiple NC files and one output CSV file
    # sys.argv[1:-1] are all the .nc file paths, sys.argv[-1] is the output CSV file path
    main()
