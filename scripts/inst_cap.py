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

def inst_cap_table(n):
    #generators
    gen = n.generators
    gen['country'] = gen['bus'].str[:2]
    gen = gen.groupby(['country','carrier']).p_nom_opt.sum()/1000 #GW
    gen = gen.unstack()
    gen = gen.drop(columns=['load'],inplace=False)
    gen.fillna(0, inplace=True)

    #storage units
    storage = n.storage_units
    storage['country'] = storage['bus'].str[:2]
    store = storage.groupby(['country','carrier']).p_nom_opt.sum()/1000 #GW
    store = store.unstack()
    store.fillna(0, inplace=True)

    #lines (AC)
    cap_AC = n.lines
    cap_AC['country'] = cap_AC['bus0'].str[:2]
    cap_AC = cap_AC.groupby(['country', 'carrier']).s_nom_opt.sum()
    cap_AC.fillna(0, inplace=True)
    cap_AC = cap_AC.unstack().div(1e3) #GVA

    #links (DC)
    cap_DC = n.links #DC
    cap_DC['country'] = cap_DC['bus0'].str[:2]
    cap_DC = cap_DC.groupby(['country', 'carrier']).p_nom_opt.sum()
    cap_DC.fillna(0, inplace=True)
    cap_DC = cap_DC.unstack().div(1e3) #GW

    #combine
    cap_tot = pd.concat([gen,store,cap_AC,cap_DC], axis = 1).fillna(0)

    return cap_tot


def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    n = pypsa.Network(input_file)
    df = inst_cap_table(n)
    df.to_csv(output_file)


if __name__ == "__main__":
    # The script takes multiple NC files and one output CSV file
    # sys.argv[1:-1] are all the .nc file paths, sys.argv[-1] is the output CSV file path
    main()
