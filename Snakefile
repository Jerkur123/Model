import os
import shutil
import yaml
from netCDF4 import Dataset


configfile: 'config/config.yaml'

# Load configuration
#with open(configfile) as f:
#    config = yaml.safe_load(f)

buses = config["buses"]
transmission_limits = config['transmission_limit']
countries = config['countries']
contingencies = config['contingencies']
models = config['models']
cut_start = config['cut_start']
sens_analysis = config['sens_analysis']
horizon = config['horizon']

# Ensure horizon is an integer
try:
    horizon = int(horizon)
except ValueError:
    raise ValueError(f"Horizon value must be an integer, got {horizon}")

# Extract sensitivity analysis values from the list
sens_analysis_enabled = sens_analysis[0]['enabled']
deviation_reductionto = sens_analysis[1]['deviation_reductionto']
deviation_duration = sens_analysis[2]['deviation_duration']

# Verify that the values are correctly extracted
print(f"Sensitivity Analysis Enabled: {sens_analysis_enabled} (Type: {type(sens_analysis_enabled)})")
print(f"Deviation Reductionto: {deviation_reductionto} (Type: {type(deviation_reductionto)})")
print(f"Deviation Duration: {deviation_duration} (Type: {type(deviation_duration)})")

# Transform the contingencies structure into a more accessible format and extend based on sensitivity analysis
contingency_list = []
for contingency in contingencies:
    for name, params in contingency.items():
        reductionto = next(item['reductionto'] for item in params if 'reductionto' in item)
        duration = next(item['duration'] for item in params if 'duration' in item)
        
        # Start with the original values
        reductiontos = [reductionto]
        durations = [duration]
        
        # Extend with sensitivity analysis values if enabled
        if sens_analysis_enabled:
            reductiontos.extend([
                max(0, round(reductionto - deviation_reductionto * reductionto, 2)),
                round(reductionto + deviation_reductionto * reductionto, 2)
            ])
            
            durations.extend([
                max(0, duration - deviation_duration * duration),
                duration + deviation_duration * duration
            ])
        
        # Sort the lists: reductionto in descending order, durations in ascending order
        reductiontos = sorted(reductiontos, reverse=True)
        durations = sorted(durations)

        for rt in reductiontos:
            for dur in durations:
                contingency_list.append({
                    'name': name,
                    'reductionto': rt,
                    'duration': int(dur)  # Convert duration to integer
                })

print("Extended Contingency list:", contingency_list)

# Create a list of dictionaries for each combination, including extended contingencies
combinations = []
for country in countries:
    for tl in transmission_limits:
        for model in models:
            for c in contingency_list:
                combinations.append({
                    'country': country,
                    'buses': buses,
                    'contingency': c['name'],
                    'reductionto': c['reductionto'],
                    'duration': int(c['duration']),  # Convert duration to integer
                    'transmission_limit': tl,
                    'model': model
                })

print(combinations)

rule all:
    input:
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
        horizon = horizon
    run:
        if not os.path.exists(input[0]):
            print(f"Input file {input[0]} does not exist. Skipping.")
            raise MissingInputException(f"Input file {input[0]} does not exist.")
        
        if not os.path.exists(output[0]) or not os.path.exists(output[1]):
            script = "scripts/solve_base.py"
            shell(f"python {script} {input[0]} {output[0]} {params.co2_price} {output[1]} {params.horizon}")
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
        cut_start = cut_start,
        horizon = horizon
    run:
        if os.path.exists(output[0]) and os.path.exists(output[1]) :
            print(f"Skipping processing for {output[0]} as it already exists. Skipping")
            return

        if wildcards.contingency == "pv":
            # Create empty .nc file for pv
            #with Dataset(output[0], 'w', format='NETCDF4') as ncfile:
            #    pass  # Creates an empty .nc file
            #with Dataset(output[1], 'w', format='NETCDF4') as ncfile:
            #    pass  # Creates an empty .nc file
            script = "scripts/solve_pv.py"
            shell(f"python {script} {input[0]} {output[0]} {output[1]} {wildcards.contingency} {wildcards.reductionto} {wildcards.duration} {wildcards.model} {params.horizon} {params.cut_start}")
        elif wildcards.contingency == "wind":
            # Create empty .nc file for pv
            #with Dataset(output[0], 'w', format='NETCDF4') as ncfile:
            #    pass  # Creates an empty .nc file
            #with Dataset(output[1], 'w', format='NETCDF4') as ncfile:
            #    pass  # Creates an empty .nc file
            script = "scripts/solve_wind.py"
            shell(f"python {script} {input[0]} {output[0]} {output[1]} {wildcards.contingency} {wildcards.reductionto} {wildcards.duration} {wildcards.model} {params.horizon} {params.cut_start}")
