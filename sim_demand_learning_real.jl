
begin
	dir = @__DIR__
	include("$dir/src/system_structs.jl")
	include("$dir/src/network_dynamics.jl")
end

begin
	using JLD2, FileIO, GraphIO, CSV, DataFrames
	using Distributed
	using Interpolations
	using DifferentialEquations
	using Distributions
	using LightGraphs
	using LinearAlgebra
	using Random
	using DSP
	using ToeplitzMatrices
	Random.seed!(42)
end

begin
	N = 4
	num_days =  35
	freq_threshold = 0.2
end

begin
	phase_filter = 1:N
	freq_filter = N+1:2N
	control_filter = 2N+1:3N
	energy_filter = 3N+1:4N
	energy_abs_filter = 4N+1:5N
end


############################################

begin
	l_day = 3600*24
	l_hour = 3600
	l_minute = 60
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=0.2,kP=52,T_inv=1/0.05,kI=10)
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=0.2,kP=525,T_inv=1/0.05,kI=0.005)
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner=N),kI=repeat([0.005], inner=N)) # different for each node, change array
	low_layer_control = system_structs.LeakyIntegratorPars(M_inv=[1/5.; 1/4.8; 1/4.1; 1/4.8],kP= [400.; 110.; 100.; 200.],T_inv=[1/0.04; 1/0.045; 1/0.047; 1/0.043],kI=[0.05; 0.004; 0.05; 0.001]) # different for each node, change array
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=[0.1; 10; 100; 1000],T_inv=repeat([1/0.05], inner=N),kI=repeat([0.005], inner=N)) # different for each node, change array
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=[1/0.05; 1/0.5; 1/5; 1/50],kI=repeat([0.005], inner=N)) # different for each node, change array
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner = N),kI=[0.005; 0.5; 5; 500]) # different for each node, change array
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=[0.002; 0.2; 2; 20],kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner = N),kI=repeat([0.005], inner=N)) # different for each node, change array
	kappa = 1. / l_hour
end

############################################
# NETWORK - this should only run on one process
############################################

# # Full graph for N=4 and degree 3 graph otherwise, change last 3 to 1 for N=2
graph = random_regular_graph(iseven(3N) ? N : (N-1), 3) # change last "3" to 1 for N=2

# N = 1
#graph = SimpleGraph(1)

# # Square - needs to be changed only here
# graph = SimpleGraph(4)
# add_edge!(_graph_lst, 1,2)
# add_edge!(_graph_lst, 2,3)
# add_edge!(_graph_lst, 3,4)
# add_edge!(_graph_lst, 4,1)


# using GraphPlot
# gplot(graph)

# # Line - needs to be changed only here
# graph = SimpleGraph(4)
# add_edge!(_graph_lst, 1,2)
# add_edge!(_graph_lst, 2,3)
# add_edge!(_graph_lst, 3,4)
# using GraphPlot
# gplot(graph)



############################################
#  demand
############################################

struct demand_amp_var
	demand
end

function (dav::demand_amp_var)(t)
	index = Int(floor(t / (24*3600)))
	dav.demand[index + 1,:]
end

# demand_amp = rand(N) .* 250. # fixed amp over the days

# # slowly increasing amplitude - only working fpr 10 days now
# demand_ampp = demand_amp_var(repeat([10 20 30 40 50 60 70 80 90 100 110], outer=Int(N/2))') # random positive amp over days by 10%
# demand_ampn = demand_amp_var(repeat([-10 -20 -30 -40 -50 -60 -70 -80 -90 -100 -110], outer=Int(N/2))') # random positive amp over days by 10%
# demand_amp = t->vcat(demand_ampp(t), demand_ampn(t))

# # slowly decreasing amplitude - only working fpr 10 days now
# demand_ampp = demand_amp_var(repeat([110 100 90 80 70 60 50 40 30 20 10], outer=Int(N/2))') # random positive amp over days by 10%
# demand_ampn = demand_amp_var(repeat([-110 -100 -90 -80 -70 -60 -50 -40 -30 -20 -10], outer=Int(N/2))') # random positive amp over days by 10%
# demand_amp = t->vcat(demand_ampp(t), demand_ampn(t))

# # real demand pattern for 5 weeks
# demand_ampp = demand_amp_var(repeat([120 120 120 120 120 170 200 120 120 120 120 120 170 200 120 120 120 120 120 170 200 120 120 120 120 120 170 200 120 120 120 120 120 170 200 120], outer=Int(N/2))') # random positive amp over days by 10%
# demand_ampn = demand_amp_var(repeat([120 120 120 120 120 170 200 120 120 120 120 120 170 200 120 120 120 120 120 170 200 120 120 120 120 120 170 200 120 120 120 120 120 170 200 120], outer=Int(N/2))') # random positive amp over days by 10%
# demand_amp = t->vcat(demand_ampp(t), demand_ampn(t))

# slowly increasing and decreasing amplitude - only working for <= 20 days now
# demand_ampp = demand_amp_var(repeat([80 80 80 10 10 10 40 40 40 40 40], outer=Int(N/2))') # random positive amp over days by 10%
# demand_ampn = demand_amp_var(repeat([80 80 80 10 10 10 40 40 40 40 40], outer=Int(N/2))') # random positive amp over days by 10%
# demand_amp = t->vcat(demand_ampp(t), demand_ampn(t))


# # random positive amp over days by 30%
# demand_ampp = demand_amp_var(70 .+ rand(num_days+1,Int(N/2)).* 30.)
# demand_ampn = demand_amp_var(70 .+ rand(num_days+1,Int(N/2)).* 30.)  # random negative amp over days by 10%
# demand_amp = t->vcat(demand_ampp(t), demand_ampn(t))

dem_data = CSV.read("$dir/profil.csv")

# using StatsPlots
# @df dem_data plot(0.25:0.25:24,:Werktag, label="Werktag Winter", legend=:topleft)
# @df dem_data plot!(0.25:0.25:24,:Samstag, label="Samstag Winter")
# @df dem_data plot!(0.25:0.25:24,:Sonntag, label="Sonntag Winter")

# weekday_winter = t->dem_data[!,:Werktag][Int(floor(mod(t,24*3600) / 900)+1)]
# weekday_summer = t->dem_data[!,:Werktag_1][Int(floor(mod(t,24*3600) / 900)+1)]
# weekday_between = t->dem_data[!,:Werktag_2][Int(floor(mod(t,24*3600) / 900)+1)]
#
# saturday_winter = t->dem_data[!,:Samstag][Int(floor(mod(t,24*3600) / 900)+1)]
# saturday_summer = t->dem_data[!,:Samstag_1][Int(floor(mod(t,24*3600) / 900)+1)]
# saturday_between = t->dem_data[!,:Samstag_2][Int(floor(mod(t,24*3600) / 900)+1)]
#
# sunday_winter = t->dem_data[!,:Sonntag][Int(floor(mod(t,24*3600) / 900)+1)]
# sunday_summer = t->dem_data[!,:Sonntag_1][Int(floor(mod(t,24*3600) / 900)+1)]
# sunday_between = t->dem_data[!,:Sonntag_2][Int(floor(mod(t,24*3600) / 900)+1)]

dem_data_week = CSV.read("$dir/profil_week.csv")

week_winter = t->dem_data_week[!,:Winterwoche][Int(floor(mod(t,24*3600*7) / 900)+1)]
week_summer = t->dem_data_week[!,:Sommerwoche][Int(floor(mod(t,24*3600*7) / 900)+1)]
week_between = t->dem_data_week[!,:Uebergangswoche][Int(floor(mod(t,24*3600*7) / 900)+1)]
week_G1 = t->dem_data_week[!,:WinterwocheG1][Int(floor(mod(t,24*3600*7) / 900)+1)]
week_G4 = t->dem_data_week[!,:WinterwocheG4][Int(floor(mod(t,24*3600*7) / 900)+1)]

week = t->vcat(week_winter(t), week_G1(t), week_G4(t), (week_winter(t) + week_G1(t) + week_G4(t))./3)


periodic_demand = t ->  week(t) ./100 #t-> demand_amp(t) .* sin(t*pi/(24*3600))^2
samples = 24*60 #24*4
inter = interpolate([10. ./100 * randn(N) for i in 1:(num_days * samples + 1)], BSpline(Linear()))
residual_demand = t -> inter(1. + t / (24*3600) * samples) # 1. + is needed to avoid trying to access out of range


#########################################
#            SIM                        #
#########################################

vc1 = 1:N # ilc_nodes (here: without communication)
cover1 = Dict([v => [] for v in vc1])# ilc_cover
u = [zeros(1000,1);1;zeros(1000,1)];
fc = 1/6;
a = digitalfilter(Lowpass(fc),Butterworth(2));
Q1 = filtfilt(a,u);#Markov Parameter
Q = Toeplitz(Q1[1001:1001+24-1],Q1[1001:1001+24-1]);


compound_pars = system_structs.compound_pars(N, low_layer_control, kappa, vc1, cover1, Q)

compound_pars.hl.daily_background_power .= 0
compound_pars.hl.current_background_power .= 0
compound_pars.hl.mismatch_yesterday .= 0.
compound_pars.periodic_demand  = periodic_demand # t -> zeros(N) #periodic_demand
compound_pars.residual_demand = residual_demand #t -> zeros(N) #residual_demand
compound_pars.graph = graph


using Plots
dd = t->((periodic_demand(t) .+ residual_demand(t)))
plot(0:7*l_day, t -> dd(t)[1],ytickfontsize=14,
               xtickfontsize=18, margin=5Plots.mm,
    		   legendfontsize=12, linewidth=3,xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)),xaxis=("days [c]",font(14)), yaxis=("normed demand",font(14)), legend=nothing)
#title!("Demand for one week in winter (household)")
savefig("$dir/plots/real_demand_winter_week.png")




@everywhere begin
	factor = 0#0.01*rand(compound_pars.D * compound_pars.N)#0.001#0.00001
	ic = factor .* ones(compound_pars.D * compound_pars.N)
	tspan = (0., num_days * l_day)
	ode_tl1 = ODEProblem(network_dynamics.ACtoymodel!, ic, tspan, compound_pars,
	callback=CallbackSet(PeriodicCallback(network_dynamics.HourlyUpdate(), l_hour),
						 PeriodicCallback(network_dynamics.DailyUpdate_X, l_day)))
end

@time sol1 = solve(ode_tl1, Rodas4())


#######################################################################
#                               PLOTTING                             #
######################################################################

using Plots
hourly_energy = zeros(24*num_days+1,N)
for i=1:24*num_days+1
	for j = 1:N
		hourly_energy[i,j] = sol1((i-1)*3600)[energy_filter[j]]
	end
end
plot(hourly_energy)

ILC_power = zeros(num_days+2,24,N)
norm_energy_d = zeros(num_days,N)
mean_energy_d = zeros(num_days,N)

for j = 1:N
	ILC_power[2,:,j] = Q*(zeros(24,1) +  kappa*hourly_energy[1:24,j])
	norm_energy_d[1,j] = norm(hourly_energy[1:24,j])
	mean_energy_d[1,j] = mean(hourly_energy[1:24,j])
end


for i=2:num_days
	for j = 1:N
		ILC_power[i+1,:,j] = Q*(ILC_power[i,:,j] +  kappa*hourly_energy[(i-1)*24+1:i*24,j])
		norm_energy_d[i,j] = norm(hourly_energy[(i-1)*24+1:i*24,j])
		mean_energy_d[i,j] = mean(hourly_energy[(i-1)*24+1:i*24,j])
	end
end

ILC_power_agg = maximum(mean(ILC_power.^2,dims=3),dims=2)
ILC_power_agg = mean(ILC_power,dims=2)
ILC_power_agg_norm = norm(ILC_power)
ILC_power_hourly = vcat(ILC_power[:,:,1]'...)

# load_amp = []
# for i = 1:num_days
# push!(load_amp, maximum.(dd(t)))
# end

load_amp_hourly = [maximum.(dd(t)) for t in 1:3600:3600*24*num_days]
using LaTeXStrings

# hourly plotting
plot(1:num_days*24, ILC_power_hourly[1:24*num_days] , legend=:topleft, label=L"$ u_j^{ILC}$", ytickfontsize=14,
               xtickfontsize=18,
    		   legendfontsize=12, linewidth=3,xaxis=("time [h]",font(14)), yaxis=("normed power",font(14)))
plot!(1:num_days*24+1,mean(hourly_energy, dims=2)/3600 , label=L"y^{c,h}", linewidth=3)
plot!(1:24*num_days, mean.(load_amp_hourly), label = "peak demand", linewidth=3)
#xlabel!("hour h [h]")
#ylabel!("normed quantities [a.u.]")
savefig("$dir/plots/real_demand_hourly_hetero.png")

# second-wise
plot(1:3600:num_days*24*3600,  ILC_power_hourly[1:num_days*24]./ maximum(ILC_power_hourly), label=L"$P_{ILC, j}$", ytickfontsize=14,
               xtickfontsize=18,
    		   legendfontsize=10, linewidth=3,xaxis=("time [s]",font(14)), yaxis=("normed quantities [a.u.]",font(14)))
plot!(1:3600:24*num_days*3600,mean(hourly_energy[1:num_days*24], dims=2) ./ maximum(hourly_energy), label=L"y_h",linewidth=3, linestyle=:dash)
dd = t->((periodic_demand(t) .+ residual_demand(t)))
plot!(0:num_days*l_day, t -> dd(t)[1], label = "demand",linewidth=3, alpha=0.3)
title!("Exemplary learning")
savefig("$dir/plots/real_demand_seconds_hetero.png")

dd = t->((periodic_demand(t) .+ residual_demand(t)))
plot(0:7*l_day, t -> dd(t)[1],ytickfontsize=20, margin=5Plots.mm,
               xtickfontsize=20,xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)),
    		   legendfontsize=12, linewidth=3,xaxis=("days [c]",font(20)), yaxis=("normed demand",font(20)), legend=nothing)
#title!("Demand for one week in winter (household)")
savefig("$dir/plots/real_demand_winter_week.png")

load_amp_hourly_N = [dd(t) for t in 1:3600:3600*24*num_days]
load_amp_hourly = sum.(load_amp_hourly_N)
load_amp_daily = sum.(mean(reshape(load_amp_hourly, 24,num_days)',dims=2))

# daily plotting
plot(2:num_days, sum(ILC_power_agg[2:num_days,1,:],dims=2), label=L"$\bar u^{ILC}$", ytickfontsize=14,
               xtickfontsize=18, margin=5Plots.mm,
    		   legendfontsize=14, linewidth=3,xaxis=("days [c]",font(14)), yaxis=("normed power",font(14)), legend=:right)
plot!(2:num_days, sum(mean_energy_d[2:num_days],dims=2) ./ 3600, label=L"\bar y^{c}", linewidth=3, linestyle=:dash)
plot!(2:num_days, load_amp_daily[2:num_days] , label = L"\bar P^d", linewidth=3, linestyle=:dashdot)
#xlabel!("day d [d]")
#ylabel!("normed quantities [a.u.]")
savefig("$dir/plots/real_demand_daily_hetero.png")
#
# plot(2:num_days, sum(ILC_power_agg_norm,dims=2), label=L"$\bar u^{ILC}$", ytickfontsize=14,
#                xtickfontsize=18, margin=5Plots.mm,
#     		   legendfontsize=14, linewidth=3,xaxis=("days [c]",font(14)), yaxis=("normed power",font(14)), legend=:right)
# plot!(2:num_days, sum(norm_energy_d[2:num_days],dims=2) ./ 3600, label=L"\bar y^{c}", linewidth=3, linestyle=:dash)
# plot!(2:num_days, load_amp_daily[2:num_days] , label = L"\bar P^d", linewidth=3, linestyle=:dashdot)
# #xlabel!("day d [d]")
# #ylabel!("normed quantities [a.u.]")
# savefig("$dir/plots/real_demand_daily_hetero_norm.png")
