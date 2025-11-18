# This code is not yet executable. Look for all comments that start and end with three hashtags and replace them by code.
# Replacing those ### comments by their code equivalent will allow you to run the code.

using JuMP
using Clp
using Plots
using DataFrames, CSV
include("helper_functions.jl")

data_path = "data"
time_series = CSV.read(joinpath(data_path, "timedata.csv"),DataFrame)
tech_data = CSV.read(joinpath(data_path, "technologies.csv"),DataFrame)
tech_data
### data preprocessing ###
T = 1:size(time_series, 1) |> collect # Save time steps in a vector
P = tech_data[:,:technology] |> Vector #Save technology types as a vector
DISP = tech_data[tech_data[!,:dispatchable] .== 1, :technology] # Create subset with dispachable
NONDISP = tech_data[tech_data[!,:dispatchable] .== 0 ,:technology] # Non discachable
S = tech_data[tech_data[!,:investment_storage] .> 0 ,:technology] #Vector of storage technologies

### parameters ###
annuity_factor(n,r) = r * (1+r)^n / (((1+r)^n)-1)
# The annuity converts the OC into equvalent uniform anual cost.
#OC is the one time investment cost per unit of capacity, as if the whole power plant was built overnight
#The power plant operates for many years, money today is woth more than money in the future
#If we only use OC we ignore that the investment pays of over many years and that the cost of capital reduces the value of future payments


interest_rate = 0.04
#We create dictionaries with (investment cost)capacities for the generation, and storage technologies, to iterate over in our model
ic_generation_cap = Dict{String, Float64}()
ic_charging_cap = Dict{String, Float64}()
ic_storage_cap = Dict{String, Float64}()
eff_in = Dict{String, Float64}()
eff_out = Dict{String, Float64}()
vc = Dict{String, Float64}() # This is variable costs (the marginal cost of production), 0 for all becaus non-dispachable
tech_data
for row in eachrow(tech_data) #iterate over rows in tech data
    af = annuity_factor(row.lifetime, interest_rate) #calculate annuity factor for each tech
    ic_generation_cap[row.technology] = row.investment_generation * af # calculate the yearly investment cost generation capacity, cost per mw

    iccc = row.investment_charge * af # yearly investment cost of charing capacity
    iccc > 0 && (ic_charging_cap[row.technology] = iccc)

    icsc = row.investment_storage * af # Yearly investment cost for storage
    icsc > 0 && (ic_storage_cap[row.technology] = icsc)

    #Here we just obtain the storage efficiencies from the data and assign it to the dict
    row.storage_efficiency_in > 0 && (eff_in[row.technology] = row.storage_efficiency_in)
    row.storage_efficiency_out > 0 && (eff_out[row.technology] = row.storage_efficiency_out)

    vc[row.technology] = row.vc
end
demand = time_series[:,:demand] |> Vector # vector for the demand series
availability = Dict(nondisp => time_series[:,nondisp] for nondisp in NONDISP)


successor(arr, x) = (x == length(arr)) ? 1 : x + 1 #function just adds one if x is the last element then it retursn to the first

#Hours per year divided by time steps in the model.
# We only have 48 hours in our model, so to make it realistic we divide a year by thise time steps
# to get it on a yearly basis
dispatch_scale = 8760/length(T)
# -> one timestep in our model corresponds to 182 hours.ø
dispatch_scale 
### model ###
m = Model(Clp.Optimizer)

@variables m begin
    # variables from our dispatch model
    G[DISP, T] >= 0 #dispach for each dispachable in each timestep
    CU[T] >= 0 #curtailment
    D_stor[S,T] >= 0 #Distcharge for each storage technology in each timeset, can be positive (stores) or negative (uses)
    L_stor[S,T] >= 0 #storage level for each S in each timestep

    # new variables for our investment model
    CAP_G[P] >= 0 #Generation capacity P represents all generation technologies 
    CAP_D[S] >= 0 #Storage carging capacity MW
    CAP_L[S] >= 0 #storage level capacity
end

@objective(m, Min, # build the objective function
    sum(### cost for dispatch ###
    vc[disp] * G[disp, t] for disp in DISP, t in T)*dispatch_scale 
    + ### cost for investment into production capactity ###
    sum(CAP_G[p]*ic_generation_cap[p] for p in P)
    + ### cost for investment into storage charging cap - use: for s in S if haskey(ic_charging_cap, s)) ###
    sum(CAP_D[s]*ic_charging_cap[s] for s in S if haskey(ic_charging_cap, s))
    + ### cost for investment into) ###
    sum(CAP_L[s]*ic_storage_cap[s]  for s in S if haskey(ic_storage_cap, s))
    )


@expression( # The amount non renewables we can use depends on how much we invest in generation capacities and their capacity factors
    m, feed_in[ndisp=NONDISP, t=T], #we therefore make a new expression for feed in
    availability[ndisp][t]*CAP_G[ndisp]
)

#The sum of dispatch and residual feed in minus curtailmant and what is stored has to satisfy the demand
@constraint(m, EnergyBalance[t in T], ### add supply-demand balance ###
    sum(G[disp, t] for disp in DISP) +
    sum(feed_in[ndisp, t] for ndisp in NONDISP) -
    CU[t] -
    sum(D_stor[s, t] for s in S)
    == demand[t]
)

#Our dispatch is limited by invested capacity
@constraint(m, GenerationCapLim[disp in DISP, t in T], ### generation capacity limit ###
 G[disp, t] <=  CAP_G[disp]

)

#if the storage technologi dosnt have its own charging capacity investment cost, then
# the model should assume that the same physical capacity is used both for charging and discharing
@constraint(m, StorageProdLim[s in S, t in T; !(haskey(ic_charging_cap,s))], ### storage production limit ###
 CAP_G[s]==CAP_D[s] #for example batteries uses the same 'hardware' for charing and discharing
 #while for seasonal_storage changing cap = electrolyxer capacity and distraching capacity = fuel cell investments
    #-> in our model, if we include storage for biogass, then i think the ladder is relevant
 )

@constraint(m, SymChargPower[t in T, s in S], ### symmetric charging power ###
    L_stor[s, successor(T, t)] == L_stor[s, t] + D_stor[s,t]*eff_in[s] - G[s, t]*(1/eff_out[s])

 #what happens when t = len(T)?
)

#The storage level at time t for each of the storage technologies cannot be higher that the capacity
@constraint(m, MaxStorageLevel[s in S, t in T], ### maximum storage level ###
 L_stor[s, t] <= CAP_L[s]
)


### storage level constraints ###
@constraint(m, MaxCharge[s in S, t in T; (haskey(ic_charging_cap, s))], ### storage production limit ###
 D_stor[s,t] <= CAP_D[s]
)

optimize!(m)

termination_status(m)

######
# Generation capacities
gen_caps = DataFrame(
    technology = P,
    capacity_MW = [value(CAP_G[p]) for p in P]
)

# storage charging capacities
store_charge_c = DataFrame(
    technology = S,
    charge_cap_mw = [value(CAP_D[s]) for s in S]
)

# === Storage level capacities (MWh) ===
stor_energy_caps = DataFrame(
    technology = S,
    storage_cap_MWh = [value(CAP_L[s]) for s in S]
)

# === Dispatch for all dispatchable technologies and timesteps ===
dispatch_df = DataFrame(
    technology = repeat(DISP, inner=length(T)),
    timestep = repeat(T, outer=length(DISP)),
    generation_MW = [value(G[disp, t]) for disp in DISP, t in T] |> vec
)


# === Feed-in from non-dispatchables (PV, wind, etc.) ===
feedin_df = DataFrame(
    technology = repeat(NONDISP, inner=length(T)),
    timestep = repeat(T, outer=length(NONDISP)),
    generation_MW = [value(feed_in[ndisp, t]) for ndisp in NONDISP, t in T] |> vec
)

objective_value(m)

dispatch_cost = sum(
    vc[disp] * value(G[disp, t]) * dispatch_scale
    for disp in DISP, t in T
)
println("Dispatch (variable) cost: ", dispatch_cost)
gen_investment_cost = sum(
    value(CAP_G[p]) * ic_generation_cap[p]
    for p in P
)
println("Generation investment cost: ", gen_investment_cost)


stor_charge_investment_cost = sum(
    value(CAP_D[s]) * ic_charging_cap[s]
    for s in S if haskey(ic_charging_cap, s)
)
println("Storage charging capacity investment cost: ", stor_charge_investment_cost)
CAP_G[s]==CAP_D[s]


stor_energy_investment_cost = sum(
    value(CAP_L[s]) * ic_storage_cap[s]
    for s in S if haskey(ic_storage_cap, s)
)
println("Storage energy capacity investment cost: ", stor_energy_investment_cost)

total_cost = dispatch_cost + gen_investment_cost + stor_charge_investment_cost + stor_energy_investment_cost
println("Total recomputed cost: ", total_cost)
println("Model objective value: ", objective_value(m))
println("Difference: ", total_cost - objective_value(m))

colordict = Dict(
    "pv" => :yellow,
    "wind" => :lightblue,
    "seasonal_storage" => :darkblue,
    "battery" => :lightgrey,
    "demand" => :darkgrey,
    "curtailment" => :red
)

######## plot electricity balance ###########

result_G = get_result(G, [:technology, :hour])
result_feed_in = get_result(feed_in, [:technology, :hour])

result_charging = get_result(D_stor, [:technology, :hour])
result_CU = get_result(CU, [:hour])
result_CU[!,:technology] .= "curtailment"
#cont here
df_demand = DataFrame(hour=T, technology="demand", value=demand)

result_generation = vcat(result_feed_in, result_G)
result_demand = vcat(result_charging, result_CU, df_demand)

table_gen = unstack(result_generation, :hour, :technology, :value)
table_gen = table_gen[!,[NONDISP..., DISP...]]
labels = names(table_gen) |> permutedims
colors = [colordict[tech] for tech in labels]
data_gen = Array(table_gen)

balance_plot = areaplot(
    data_gen,
    label=labels,
    color=colors,
    width=0,
    leg=:outertopright
)

table_dem = unstack(result_demand, :hour, :technology, :value)
table_dem = table_dem[!,["demand", S...,"curtailment"]]
labels2 = names(table_dem) |> permutedims
colors2 = [colordict[tech] for tech in labels2]
replace!(labels2, [item => "" for item in intersect(labels2, labels)]...)
data_dem = -Array(table_dem)

areaplot!(
    balance_plot,
    data_dem,
    label=labels2,
    color=colors2,
    width=0,
    leg=:outertopright
)

hline!(balance_plot, [0], color=:black, label="", width=2)

#################################

df_installed_gen = get_result(CAP_G, [:technology])
x = df_installed_gen[!,:technology]
y = df_installed_gen[!,:value] ./ 1000
p1 = bar(
    x,
    y,
    leg=false,
    title="Installed power generation",
    ylabel="GW",
    guidefontsize=8,
    rotation=45
)

df_installed_charge = get_result(CAP_D, [:technology])
x = df_installed_charge[!,:technology]
y = df_installed_charge[!,:value] ./ 1000
p2 = bar(
    x,
    y,
    leg=false,
    title="Installed power charging",
    ylim=ylims(p1),
    rotation=45
)

df_installed_storage = get_result(CAP_L, [:technology])
x = df_installed_storage[!,:technology]
y = df_installed_storage[!,:value] ./ 1e6

p3 = bar(
    x,
    y,
    leg=false,
    title="Installed storage capacity",
    ylabel="TWh",
    guidefontsize=8,
    rotation=45
)

plot(
    p1,
    p2,
    p3,
    layout=(1,3),
    titlefontsize=8,
    tickfontsize=6
)
