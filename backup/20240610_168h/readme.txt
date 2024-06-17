Contingency Parameters (config):
co2_price: 130 # €/t

countries:
 - DE:
   - bus: 20
 - GB:
   - bus: 13
 - ES:
   - bus: 8

transmission_limit:
 - 1.0
 - 1.5

contingencies: 
 - pv:
   - reductionto: 0.25
   - duration: 180
   - scenario explanation: 
	- pmaxpu of solar generators in all countries in the system are reduced in the defined period
   - allow investment method:
	- Gen, Storage Unit: Allow extendable unit to be extended
	- Lines, Links: Allow extension of all DC and AC lines

 - wind:
   - reductionto: 0.5
   - duration: 180
   - scenario explanation: 
	- pmaxpu of wind generators (on& offshore) in all countries in the system are reduced in the defined period
   - allow investment method: 
	- Gen, Storage Unit: Allow extendable unit to be extended
	- Lines, Links: Allow extension of all DC and AC lines

 - noexim:
   - reductionto: 0.25
   - duration: 180
   - scenario explanation: 
	- pmaxpu of international connection (AC & DC) are reduced in the defined period
   - allow investment method: 
	- Gen, Storage Unit: Allow extendable unit to be extended
	- Lines, Links: only national lines are extendable (row['bus0'][:2] == row['bus1'][:2])	

 - drought:
   - reductionto: 0.5
   - duration: 180
   - scenario explanation:
	- pmaxpu of ror, hydro and nucelar in all countries in the system are reduced in the defined period
   - allow investment method: 
	- Gen, Storage Unit: Allow extendable unit to be extended
	- Lines, Links: Allow extension of all DC and AC lines

 - dispatchcut:
   - reductionto: 0.25
   - duration: 180
   - scenario explanation:
	- generator other than 'coal','lignite','load','solar','onwind','offwind-dc','offwind-ac'
	  or hydro or phs with the biggest installed capacity in the selected country only (neighbors excluded)
	  is reduced in the defined period
   - allow investment method:
	- Gen, Storage Unit: Allow extendable units in the selected country to be expanded (neighbors excluded)
	- Lines, Links: Allow connection with the main country in 'bus0' or 'bus1' are extendable


cut_start: '2013-02-01 00:00:00'

models:
 - inv
 - noinv

horizon: 1 #in number of days*8 because the of the 3H time resolution

sens_analysis:
 - enabled: true
 - deviation_reductionto: 0.25 
 - deviation_duration: 90

H2_cost_sens_analysis:
 - enabled: false
 - deviation: 0.2 # %


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



Allow investment method:
- pv, wind, drought:
		- Gen, Storage Unit: Allow extendable unit to be extended
		- Lines, Links: Allow extension of all DC and AC lines
-noexim:
		- Gen, Storage Unit: Allow extendable unit to be extended
		- Lines, Links: only national lines are extendable (row['bus0'][:2] == row['bus1'][:2])

No Investment method:
- all contingency scenario:
		- Gen, Storage Units, Lines, Links: set all extendable switch to false