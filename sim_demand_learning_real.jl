using JLD2, FileIO, GraphIO, CSV, DataFrames
using Distributed
using Interpolations

_calc = false
_slurm = false

if _calc
    using ClusterManagers
	if length(ARGS) > 0
		N_tasks = parse(Int, ARGS[1])
	else
		N_tasks = 1
	end
    N_worker = N_tasks
	if _slurm
    	addprocs(SlurmManager(N_worker))
	else
		addprocs(N_worker)
	end
	println()
	println(nprocs(), " processes")
	println(length(workers()), " workers")
else
	using Plots
end

# here comes the broadcast
# https://docs.julialang.org/en/v1/stdlib/Distributed/index.html#Distributed.@everywhere
@everywhere begin
	calc = $_calc # if false, only plotting
end

@everywhere begin
	dir = @__DIR__
	#include("$dir/exp_base.jl")
	include("$dir/src/experiments.jl")
#	include("$dir/input_data/demand_curves.jl")
	include("$dir/src/network_dynamics.jl")
end

@everywhere begin
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
	batch_size = 1
end

@everywhere begin
	freq_threshold = 0.2
	phase_filter = 1:N
	freq_filter = N+1:2N
	control_filter = 2N+1:3N
	energy_filter = 3N+1:4N
	energy_abs_filter = 4N+1:5N
end


############################################

@everywhere begin
	l_day = 3600*24 # DemCurve.l_day
	l_hour = 3600 # DemCurve.l_hour
	l_minute = 60 # DemCurve.l_minute
	#low_layer_control = experiments.LeakyIntegratorPars(M_inv=0.2,kP=52,T_inv=1/0.05,kI=10)
	#low_layer_control = experiments.LeakyIntegratorPars(M_inv=0.2,kP=525,T_inv=1/0.05,kI=0.005)
	#low_layer_control = experiments.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner=N),kI=repeat([0.005], inner=N)) # different for each node, change array
	low_layer_control = experiments.LeakyIntegratorPars(M_inv=[1/5.; 1/4.8; 1/4.1; 1/4.8],kP= [400.; 110.; 100.; 200.],T_inv=[1/0.04; 1/0.045; 1/0.047; 1/0.043],kI=[0.05; 0.004; 0.05; 0.001]) # different for each node, change array
	#low_layer_control = experiments.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=[0.1; 10; 100; 1000],T_inv=repeat([1/0.05], inner=N),kI=repeat([0.005], inner=N)) # different for each node, change array
	#low_layer_control = experiments.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=[1/0.05; 1/0.5; 1/5; 1/50],kI=repeat([0.005], inner=N)) # different for each node, change array
	#low_layer_control = experiments.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner = N),kI=[0.005; 0.5; 5; 500]) # different for each node, change array
	#low_layer_control = experiments.LeakyIntegratorPars(M_inv=[0.002; 0.2; 2; 20],kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner = N),kI=repeat([0.005], inner=N)) # different for each node, change array
	kappa = 1. / l_hour
end

############################################
# this should only run on one process
############################################

# # Full graph for N=4 and degree 3 graph otherwise, change last 3 to 1 for N=2
_graph_lst = []
for i in 1:1
	push!(_graph_lst, random_regular_graph(iseven(3N) ? N : (N-1), 3)) # change last "3" to 1 for N=2
end
@everywhere graph_lst = $_graph_lst

# N = 1
#graph_lst = [SimpleGraph(1)]

# # Square - needs to be changed only here
# _graph_lst = SimpleGraph(4)
# add_edge!(_graph_lst, 1,2)
# add_edge!(_graph_lst, 2,3)
# add_edge!(_graph_lst, 3,4)
# add_edge!(_graph_lst, 4,1)
# _graph_lst = [_graph_lst]
# @everywhere graph_lst = $_graph_lst


# using GraphPlot
# gplot(graph_lst[1])

# # Line - needs to be changed only here
# _graph_lst = SimpleGraph(4)
# add_edge!(_graph_lst, 1,2)
# add_edge!(_graph_lst, 2,3)
# add_edge!(_graph_lst, 3,4)
# _graph_lst = [_graph_lst]
# @everywhere graph_lst = $_graph_lst
# using GraphPlot
# gplot(graph_lst[1])

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

#demand_amp = rand(N) .* 250. # fixed amp over the days
# demand_ramp = rand(N) .* 2. # does not work

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
# plot(0:0.01:8*24*3600, t -> periodic_demand(t)[2])
samples = 24*60 #24*4
inter = interpolate([10. ./100 * randn(N) for i in 1:(num_days * samples + 1)], BSpline(Linear()))
residual_demand = t -> inter(1. + t / (24*3600) * samples) # 1. + is needed to avoid trying to access out of range
# f = t -> compound_pars.residual_demand(t) .+ compound_pars.periodic_demand(t)
# plot(1:100000, t -> f(t)[1])

#demand = [DemCurve.get_random_day_seq(data,num_days*10) |> DemCurve.interp_data for n in 1:N] # does it need to depend on "run" somehow?
#compound_pars.residual_demand = t -> [d(t) for d in demand]



#########################################
#            SIM 1                     #
#########################################
#my = [0 0.0005 -0.0019 -0.0002 0.003 -0.0011 -0.0009 -0.0017 0.0065 -0.0039 -0.0001 -0.0055 0.0151 0.0054 0.0038 -0.0109 -0.0010 -0.0017 0.0103 -0.0053 0.0032 -0.0136 -0.0411 -0.7053]
#my = zeros(1,24)

vc1 = 1:N # ilc_nodes (here: without communication)
cover1 = Dict([v => [] for v in vc1])# ilc_cover
u = [zeros(1000,1);1;zeros(1000,1)];
fc = 1/6;
a = digitalfilter(Lowpass(fc),Butterworth(2));
Q1 = filtfilt(a,u);#Markov Parameter
Q = Toeplitz(Q1[1001:1001+24-1],Q1[1001:1001+24-1]);

# kappa_lst = (0:0.01:2) ./ l_hour
# kappa_lst = (0:.25:2) ./ l_hour
# kappa = kappa_lst[1]
#num_monte = batch_size*length(kappa_lst)

_compound_pars = experiments.compound_pars(N, low_layer_control, kappa, vc1, cover1, Q)

_compound_pars.hl.daily_background_power .= 0#0.001
_compound_pars.hl.current_background_power .= 0# 0.001
_compound_pars.hl.mismatch_yesterday .= 0. #[my;-my]'
_compound_pars.periodic_demand  = periodic_demand # t -> zeros(N) #periodic_demand
_compound_pars.residual_demand = residual_demand #t -> zeros(N) #residual_demand
_compound_pars.graph = graph_lst[1]

@everywhere compound_pars = $_compound_pars


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


hourly_energy = zeros(24*num_days+1,N)
for i=1:24*num_days+1
	hourly_energy[i,1] = sol1((i-1)*3600)[energy_filter[1]]
	hourly_energy[i,2] = sol1((i-1)*3600)[energy_filter[2]]
	hourly_energy[i,3] = sol1((i-1)*3600)[energy_filter[3]]
	hourly_energy[i,4] = sol1((i-1)*3600)[energy_filter[4]]
end
plot(hourly_energy)

ILC_power = zeros(num_days+2,24,N)
ILC_power[2,:,1] = Q*(zeros(24,1) +  kappa*hourly_energy[1:24,1])
ILC_power[2,:,2] = Q*(zeros(24,1) +  kappa*hourly_energy[1:24,2])
ILC_power[2,:,3] = Q*(zeros(24,1) +  kappa*hourly_energy[1:24,3])
ILC_power[2,:,4] = Q*(zeros(24,1) +  kappa*hourly_energy[1:24,4])
norm_energy_d = zeros(num_days,N)
norm_energy_d[1,1] = norm(hourly_energy[1:24,1])
norm_energy_d[1,2] = norm(hourly_energy[1:24,2])
norm_energy_d[1,3] = norm(hourly_energy[1:24,3])
norm_energy_d[1,4] = norm(hourly_energy[1:24,4])
mean_energy_d = zeros(num_days,N)
mean_energy_d[1,1] = mean(hourly_energy[1:24,1])
mean_energy_d[1,2] = mean(hourly_energy[1:24,2])
mean_energy_d[1,3] = mean(hourly_energy[1:24,3])
mean_energy_d[1,4] = mean(hourly_energy[1:24,4])


for i=2:num_days
	ILC_power[i+1,:,1] = Q*(ILC_power[i,:,1] +  kappa*hourly_energy[(i-1)*24+1:i*24,1])
	ILC_power[i+1,:,2] = Q*(ILC_power[i,:,2] +  kappa*hourly_energy[(i-1)*24+1:i*24,2])
	ILC_power[i+1,:,3] = Q*(ILC_power[i,:,3] +  kappa*hourly_energy[(i-1)*24+1:i*24,3])
	ILC_power[i+1,:,4] = Q*(ILC_power[i,:,4] +  kappa*hourly_energy[(i-1)*24+1:i*24,4])
	norm_energy_d[i,1] = norm(hourly_energy[(i-1)*24+1:i*24,1])
	norm_energy_d[i,2] = norm(hourly_energy[(i-1)*24+1:i*24,2])
	norm_energy_d[i,3] = norm(hourly_energy[(i-1)*24+1:i*24,3])
	norm_energy_d[i,4] = norm(hourly_energy[(i-1)*24+1:i*24,4])
	mean_energy_d[i,1] = mean(hourly_energy[(i-1)*24+1:i*24,1])
	mean_energy_d[i,2] = mean(hourly_energy[(i-1)*24+1:i*24,2])
	mean_energy_d[i,3] = mean(hourly_energy[(i-1)*24+1:i*24,3])
	mean_energy_d[i,4] = mean(hourly_energy[(i-1)*24+1:i*24,4])
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

plot(2:num_days, sum(ILC_power_agg_norm[2:num_days,1,:],dims=2), label=L"$\bar u^{ILC}$", ytickfontsize=14,
               xtickfontsize=18, margin=5Plots.mm,
    		   legendfontsize=14, linewidth=3,xaxis=("days [c]",font(14)), yaxis=("normed power",font(14)), legend=:right)
plot!(2:num_days, sum(norm_energy_d[2:num_days],dims=2) ./ 3600, label=L"\bar y^{c}", linewidth=3, linestyle=:dash)
plot!(2:num_days, load_amp_daily[2:num_days] , label = L"\bar P^d", linewidth=3, linestyle=:dashdot)
#xlabel!("day d [d]")
#ylabel!("normed quantities [a.u.]")
savefig("$dir/plots/real_demand_daily_hetero_norm.png")



#sol2 = CSV.read("$dir/files/test2.csv", header=false)
#plot(sol2[:Column1])

#p1 = plot()
#plot(sol2[Symbol("0.0_9")])




# hourly_energy = zeros(24*num_days,N)
#
# for i=1:24*num_days
# 	hourly_energy[i,1] = sol1(i*3600)[energy_filter[1]]
# 	hourly_energy[i,2] = sol1(i*3600)[energy_filter[2]]
# 	hourly_energy[i,3] = sol1(i*3600)[energy_filter[3]]
# 	hourly_energy[i,4] = sol1(i*3600)[energy_filter[4]]
# end

# plot(hourly_energy)
# savefig("$dir/plots/hourly_energy_for_comparison.png")

#
# norm_energy = zeros(num_days,N)
# norm_energy_d = zeros(num_days,N)
#
#
# for i=1:num_days
# 	# norm_energy[i,1] = norm(sol1((i-1)*3600*24+1:i*3600*24)[energy_filter[1]])
# 	# norm_energy[i,2] = norm(sol1((i-1)*3600*24+1:i*3600*24)[energy_filter[2]])
# 	# norm_energy[i,3] = norm(sol1((i-1)*3600*24+1:i*3600*24)[energy_filter[3]])
# 	# norm_energy[i,4] = norm(sol1((i-1)*3600*24+1:i*3600*24)[energy_filter[4]])
# 	norm_energy_d[i,1] = norm(hourly_energy[(i-1)*24+1:i*24,1])
# 	norm_energy_d[i,2] = norm(hourly_energy[(i-1)*24+1:i*24,2])
# 	norm_energy_d[i,3] = norm(hourly_energy[(i-1)*24+1:i*24,3])
# 	norm_energy_d[i,4] = norm(hourly_energy[(i-1)*24+1:i*24,4])
# end
#
# # plot(norm_energy)
# plot(norm_energy_d, label=["node 1","node 2", "node 3", "node 4"])
# ylabel!("daily p2 norm of the hourly integral of PLI [Ws]")
# xlabel!("days")
# title!("Square network with kappa = 1.5")
# savefig("$dir/plots/P_LI_norm_over_days_square_kappa1-5.png")
#
# plot(sol1, vars = energy_filter)
# savefig("$dir/plots/simI_control_Xiaohan_kp525_ki0005_N4_pn_in-decrease_Q_Ysquare.png")
#
# plot(mod2pi.([p[phase_filter[1]] for p in sol1.u]))



# # never save the solutions INSIDE the git repo, they are too large, please make a folder solutions at the same level as the git repo and save them there
# jldopen("../../solutions/sol_def_N4.jld2", true, true, true, IOStream) do file
# 	file["sol1"] = sol1
# end
#
# @save "../../solutions/sol_kp525_ki0005_N4_pn_de-in_Q.jld2" sol1


# plot(sol1, vars = freq_filter)
# savefig("$dir/plots/simI_control_Xiaohan_kp525_ki0005_N4_pn_de-increase_Q_Ysquare.png")
#
# energy_h = zeros(num_days*24)
# for h = 1:num_days*24
# 	energy_h[h] = sol1(d*3600)[energy_filter][1]
# end
#
# energy = zeros(num_days)
# for d in 1:num_days
# 	energy[d] = sum(energy_h[(d-1)*24+1:d*24])
# end
#
# plot(1:num_days*24,energy_h, seriestype = :scatter, label = ["hourly integrated P_LI"])
# savefig("$dir/plots/P_LI.png")
#
# plot(1:num_days,energy, seriestype = :scatter, label = ["hourly integrated P_LI summed over day"])
# savefig("$dir/plots/P_LI_summed_over_day.png")
#
#
# dd = t->(periodic_demand(t) .+ residual_demand(t))
# plot(0:num_days*l_day, t -> dd(t)[1])
# savefig("$dir/plots/exemplary_demand.png")
#
