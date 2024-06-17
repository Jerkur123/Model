import os
import shutil
import yaml
from netCDF4 import Dataset


configfile: 'config/config.yaml'

# Load configuration
#with open(configfile) as f:
#    config = yaml.safe_load(f)

transmission_limits = config['transmission_limit']
countries = config['countries']
contingencies = config['contingencies']
models = config['models']
#cut_start = config['cut_start']
#sens_analysis = config['sens_analysis']
horizon = config['horizon']
o = config['min_equity']

# Ensure horizon is an integer
try:
    horizon = int(horizon)
except ValueError:
    raise ValueError(f"Horizon value must be an integer, got {horizon}")

# Extract sensitivity analysis values from the list
#sens_analysis_enabled = sens_analysis[0]['enabled']
#deviation_reductionto = sens_analysis[1]['deviation_reductionto']
#deviation_duration = sens_analysis[2]['deviation_duration']

# Verify that the values are correctly extracted
#print(f"Sensitivity Analysis Enabled: {sens_analysis_enabled} (Type: {type(sens_analysis_enabled)})")
#print(f"Deviation Reductionto: {deviation_reductionto} (Type: {type(deviation_reductionto)})")
#print(f"Deviation Duration: {deviation_duration} (Type: {type(deviation_duration)})")

# Transform the contingencies structure into a more accessible format and extend based on sensitivity analysis
print(contingencies)
contingency_list = []
for contingency in contingencies:
    for name, params in contingency.items():
        reductiontos = []
        durations = []
        for param in params:
            if 'reductionto' in param:
                reductiontos = param['reductionto']
            if 'duration' in param:
                durations = param['duration']
        #durations = params['duration']
        
        for rt in reductiontos:
            for dur in durations:
                contingency_list.append({
                    'name': name,
                    'reductionto': rt,
                    'duration': int(dur)  # Convert duration to integer
                })

print("Extended Contingency list:", contingency_list)
print(countries)
# Create a list of dictionaries for each combination, including extended contingencies
combinations = []
for country in countries:
    for country_code, country_datas in country.items():
        for country_data in country_datas:
            if 'bus' in country_data:
                #bus = next(item['bus'] for item in country_data if 'bus' in item)
                bus = country_data['bus']
        for tl in transmission_limits:
            for model in models:
                for c in contingency_list:
                    combinations.append({
                        'country': country_code,
                        'buses': bus,
                        'contingency': c['name'],
                        'reductionto': c['reductionto'],
                        'duration': int(c['duration']),  # Convert duration to integer
                        'transmission_limit': tl,
                        'model': model
                    })

#print(combinations)

# Dictionary to hold the second values of reductionto and duration for each contingency
second_values = {}

# Parsing each contingency
for contingency_dict in contingencies:
    for contingency, params in contingency_dict.items():
        reductionto_second = None
        duration_second = None
        for param in params:
            if 'reductionto' in param:
                reductionto_second = param['reductionto'][1]  # Access the second value
            if 'duration' in param:
                duration_second = param['duration'][1]  # Access the second value
        second_values[contingency] = {
            'reductionto': reductionto_second,
            'duration': duration_second
        }

print(second_values)



rule all:
    input:
        #capacity of investment model for each scenarios
        expand(
            "results/{country}_{buses}_{transmission_limit}_{contingency}_{reductionto}_{duration}_{model}roll.nc", 
            zip,
            country=[comb['country'] for comb in combinations],
            buses=[comb['buses'] for comb in combinations],
            contingency=[comb['contingency'] for comb in combinations],
            reductionto=[comb['reductionto'] for comb in combinations],
            duration=[comb['duration'] for comb in combinations],
            transmission_limit=[comb['transmission_limit'] for comb in combinations],
            model=[comb['model'] for comb in combinations]
        )

rule solve_base:
    input:
        "resources/{country}_{buses}_{transmission_limit}_base.nc"
    output:
        "resources/{country}_{buses}_{transmission_limit}_base_solved.nc",
        "resources/{country}_{buses}_{transmission_limit}_base_roll_solved.nc"
    params:
        co2_price = config['co2_price'],
        horizon = horizon,
        o = o
    run:
        if not os.path.exists(input[0]):
            print(f"Input file {input[0]} does not exist. Skipping.")
        else:
            if not os.path.exists(output[0]) or not os.path.exists(output[1]):
                script = "scripts/solve_base.py"
                shell(f"python {script} {input[0]} {output[0]} {params.co2_price} {output[1]} {params.horizon} {params.o} {wildcards.country} {wildcards.transmission_limit} {wildcards.buses}")
            else:
                print(f"Skipping processing for {output[0]} and {output[1]} as they already exist.")

rule dynamic_solve:
    input:
        "resources/{country}_{buses}_{transmission_limit}_base_solved.nc"
    output:
        "resources/{country}_{buses}_{transmission_limit}_{contingency}_{reductionto}_{duration}_{model}.nc",
        "results/{country}_{buses}_{transmission_limit}_{contingency}_{reductionto}_{duration}_{model}roll.nc"
    params:
        #reductionto=lambda wildcards: next(c['reductionto'] for c in contingency_list if c['name'] == wildcards.contingency),
        #duration=lambda wildcards: next(c['duration'] for c in contingency_list if c['name'] == wildcards.contingency),
        #art,
        horizon = horizon,
        o = o
    run:
        
        if not os.path.exists(input[0]):
            print(f"Input file {input[0]} does not exist. Skipping.")
        else:
            if os.path.exists(output[0]) and os.path.exists(output[1]): 
                print(f"Skipping processing for {output[0]} as it already exists. Skipping")
            else:
                #print(f'country: {wildcards.country}')
                if wildcards.contingency == "pv":
                    #Create empty .nc file for pv
                    #with Dataset(output[0], 'w', format='NETCDF4') as ncfile:
                        #pass  # Creates an empty .nc file
                    #with Dataset(output[1], 'w', format='NETCDF4') as ncfile:
                        #pass  # Creates an empty .nc file
                    script = "scripts/solve_pv.py"
                    shell(f"python {script} {input[0]} {output[0]} {output[1]} {wildcards.contingency} {wildcards.reductionto} {wildcards.duration} {wildcards.model} {params.horizon} {wildcards.country} {params.o} {wildcards.transmission_limit} {wildcards.buses}")
                elif wildcards.contingency == "wind":
                    #Create empty .nc file for wind
                    #with Dataset(output[0], 'w', format='NETCDF4') as ncfile:
                        #pass  # Creates an empty .nc file
                    #with Dataset(output[1], 'w', format='NETCDF4') as ncfile:
                        #pass  # Creates an empty .nc file
                    script = "scripts/solve_wind.py"
                    shell(f"python {script} {input[0]} {output[0]} {output[1]} {wildcards.contingency} {wildcards.reductionto} {wildcards.duration} {wildcards.model} {params.horizon} {wildcards.country} {params.o} {wildcards.transmission_limit} {wildcards.buses}")
                elif wildcards.contingency == "noexim":
                    #Create empty .nc file for noexim
                    #with Dataset(output[0], 'w', format='NETCDF4') as ncfile:
                        #pass  # Creates an empty .nc file
                    #with Dataset(output[1], 'w', format='NETCDF4') as ncfile:
                        #pass  # Creates an empty .nc file
                    script = "scripts/solve_noexim.py"
                    shell(f"python {script} {input[0]} {output[0]} {output[1]} {wildcards.contingency} {wildcards.reductionto} {wildcards.duration} {wildcards.model} {params.horizon} {wildcards.country} {params.o} {wildcards.transmission_limit} {wildcards.buses}")
                elif wildcards.contingency == "drought":
                    #Create empty .nc file for drought
                    #with Dataset(output[0], 'w', format='NETCDF4') as ncfile:
                        #pass  # Creates an empty .nc file
                    #with Dataset(output[1], 'w', format='NETCDF4') as ncfile:
                        #pass  # Creates an empty .nc file
                    script = "scripts/solve_drought.py"
                    shell(f"python {script} {input[0]} {output[0]} {output[1]} {wildcards.contingency} {wildcards.reductionto} {wildcards.duration} {wildcards.model} {params.horizon} {wildcards.country} {params.o} {wildcards.transmission_limit} {wildcards.buses}") 
                elif wildcards.contingency == "dispatchcut":
                    #Create empty .nc file for drought
                    #with Dataset(output[0], 'w', format='NETCDF4') as ncfile:
                        #pass  # Creates an empty .nc file
                    #with Dataset(output[1], 'w', format='NETCDF4') as ncfile:
                        #pass  # Creates an empty .nc file
                    script = "scripts/solve_dispatchcut.py"
                    shell(f"python {script} {input[0]} {output[0]} {output[1]} {wildcards.contingency} {wildcards.reductionto} {wildcards.duration} {wildcards.model} {params.horizon} {wildcards.country} {params.o} {wildcards.transmission_limit} {wildcards.buses}")  
                elif wildcards.contingency == "windpv":
                    #Create empty .nc file for drought
                    #with Dataset(output[0], 'w', format='NETCDF4') as ncfile:
                        #pass  # Creates an empty .nc file
                    #with Dataset(output[1], 'w', format='NETCDF4') as ncfile:
                        #pass  # Creates an empty .nc file
                    script = "scripts/solve_windpv.py"
                    shell(f"python {script} {input[0]} {output[0]} {output[1]} {wildcards.contingency} {wildcards.reductionto} {wildcards.duration} {wildcards.model} {params.horizon} {wildcards.country} {params.o} {wildcards.transmission_limit} {wildcards.buses}")         

#rule inst_cap_table:
#    input:
#        "resources/{country}_{buses}_{transmission_limit}_{contingency}_{reductionto}_{duration}_{model}.nc"
#    output:
#        "results/data/{country}/{contingency}/cap_{country}_{buses}_{transmission_limit}_{contingency}_{reductionto}_{duration}_{model}.csv"
#    run:
#        script = "scripts/inst_cap.py"
#        shell(f"python {script} {input} {output}") 

#rule gen_table:
#    input:
#        "resources/{country}_{buses}_{transmission_limit}_{contingency}_{reductionto}_{duration}_{model}roll.nc"
#    output:
#        "results/data/{country}/{contingency}/gen_{country}_{buses}_{transmission_limit}_{contingency}_{reductionto}_{duration}_{model}roll.csv"
#    run:
#        script = "scripts/gen_table.py"
#        shell(f"python {script} {input} {output}")

#rule inst_cap_base:
#    input:
#        "resources/{country}_{buses}_{transmission_limit}_base_solved.nc"
#    output:
#        "results/data/{country}/cap_{country}_{buses}_{transmission_limit}_base.csv"
#    run:
#        script = "scripts/inst_cap.py"
#        shell(f"python {script} {input} {output}") 

#rule syst_cost:
#    input:
#        "resources/{country}_{buses}_{transmission_limit}_{contingency}_{reductionto}_{duration}_{model}roll.nc"
#    output:
#        "results/data/{country}/syscost_{country}_{buses}_{transmission_limit}_{contingency}_{reductionto}_{duration}_{model}roll_country.csv",
#        "results/data/{country}/syscost_{country}_{buses}_{transmission_limit}_{contingency}_{reductionto}_{duration}_{model}roll_total.csv"
#    run:
#        script = "scripts/syst_cost.py"
#        shell(f"python {script} {input} {wildcards.country} {output[0]} {output[1]}") 

#rule inv_cap_country:
#    input:
#        base="results/data/{country}/cap_{country}_{buses}_{transmission_limit}_base.csv",
#        #contingecies = "results/data/{country}/{contingency}/cap_{country}_{buses}_{transmission_limit}_{contingency}_{reductionto}_{duration}_{model}.csv"
#        contingencies=lambda wildcards: expand(
#            "results/data/{country}/{contingency}/cap_{country}_{buses}_{transmission_limit}_{contingency}_{reductionto}_{duration}_inv.csv",
#            country=[c['country'] for c in combinations if c['buses'] == wildcards.buses and c['transmission_limit'] == wildcards.transmission_limit],
#            contingency=[c['contingency'] for c in combinations],
#            reductionto=[second_values[contingency]['reductionto'] for contingency in second_values],
#            duration=[second_values[contingency]['duration'] for contingency in second_values]
##        )
 #   output:
 ###       "results/data/{country}/inv_cap_{country}_{buses}_{transmission_limit}_summary.csv"
 #   shell:
 #       """
 #       type nul > {output}

#        """

