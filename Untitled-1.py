contingencies = [
    {'pv': [{'reductionto': [0, 0.25, 0.5]}, {'duration': [90, 180, 270]}]},
    {'wind': [{'reductionto': [0.25, 0.5, 0.75]}, {'duration': [90, 180, 270]}]},
    {'noexim': [{'reductionto': [0, 0.25, 0.5]}, {'duration': [90, 180, 270]}]},
    {'drought': [{'reductionto': [0.25, 0.5, 0.75]}, {'duration': [90, 180, 270]}]},
    {'dispatchcut': [{'reductionto': [0, 0.25, 0.5]}, {'duration': [90, 180, 270]}]}
]

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


#apple = market["fruits"]["apple"]
#print(apple)


