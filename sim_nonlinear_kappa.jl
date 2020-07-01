using JLD2, FileIO, GraphIO, CSV, DataFrames
using Distributed
using Interpolations

_calc = true
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
	include("$dir/src/system_structs.jl")
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

@everywhere begin
	N = 4
	num_days =  20
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
	l_hour = 3600 # DemCurve.update
	update = l_hour/4 #/2 for half # DemCurve.update
	n_updates_per_day=Int(floor(l_day/update)) # 24 l_day / update
	l_minute = 60 # DemCurve.l_minute
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=0.2,kP=52,T_inv=1/0.05,kI=10)
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=0.2,kP=525,T_inv=1/0.05,kI=0.005)
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner=N),kI=repeat([0.005], inner=N)) # different for each node, change array
	low_layer_control = system_structs.LeakyIntegratorPars(M_inv=[1/5.; 1/4.8; 1/4.1; 1/4.8],kP= [400.; 110.; 100.; 200.],T_inv=[1/0.04; 1/0.045; 1/0.047; 1/0.043],kI=[0.05; 0.004; 0.05; 0.001]) # different for each node, change array
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=[0.1; 10; 100; 1000],T_inv=repeat([1/0.05], inner=N),kI=repeat([0.005], inner=N)) # different for each node, change array
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=[1/0.05; 1/0.5; 1/5; 1/50],kI=repeat([0.005], inner=N)) # different for each node, change array
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner = N),kI=[0.005; 0.5; 5; 500]) # different for each node, change array
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=[0.002; 0.2; 2; 20],kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner = N),kI=repeat([0.005], inner=N)) # different for each node, change array
	kappa = 0.75 / update
end

############################################
# this should only run on one process
############################################

# # Full graph for N=4 and degree 3 graph otherwise, change last 3 to 1 for N=2
# Notation like this is easier to adapt for EnsembleProblems
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

@everywhere struct demand_amp_var
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

# # slowly decreasing and increasing amplitude - only working fpr 10 days now
# demand_ampp = demand_amp_var(repeat([110 100 90 80 70 60 50 60 70 80 90], outer=Int(N/2))') # random positive amp over days by 10%
# demand_ampn = demand_amp_var(repeat([-110 -100 -90 -80 -70 -60 -50 -60 -70 -80 -90], outer=Int(N/2))') # random positive amp over days by 10%
# demand_amp = t->vcat(demand_ampp(t), demand_ampn(t))

# slowly increasing and decreasing amplitude - only working fpr 10 days now
# demand_ampp = demand_amp_var(repeat([10 10 10 10 10 80 80 80 80 80 80 10 10 10 10 10 40 40 40 40 40], outer=Int(N/2))') # random positive amp over days by 10%
# demand_ampn = demand_amp_var(repeat([10 10 10 10 10 80 80 80 80 80 80 10 10 10 10 10 40 40 40 40 40], outer=Int(N/2))') # random positive amp over days by 10%
# demand_amp = t->vcat(demand_ampp(t), demand_ampn(t))
#

# # random positive amp over days by 30%
# demand_ampp = demand_amp_var(70 .+ rand(num_days+1,Int(N/2)).* 30.)
# demand_ampn = demand_amp_var(70 .+ rand(num_days+1,Int(N/2)).* 30.)  # random negative amp over days by 10%
# demand_amp = t->vcat(demand_ampp(t), demand_ampn(t))

demand_amp1 = demand_amp_var(60 .+ rand(num_days+1,Int(N/4)).* 40.)
demand_amp2 = demand_amp_var(70 .+ rand(num_days+1,Int(N/4)).* 30.)
demand_amp3 = demand_amp_var(80 .+ rand(num_days+1,Int(N/4)).* 20.)
demand_amp4 = demand_amp_var(90 .+ rand(num_days+1,Int(N/4)).* 10.)
demand_amp = t->vcat(demand_amp1(t), demand_amp2(t),demand_amp3(t),demand_amp4(t))



periodic_demand =  t-> demand_amp(t)./100 .* sin(t*pi/(24*3600))^2
samples = 24*4
inter = interpolate([.2 * randn(N) for i in 1:(num_days * samples + 1)], BSpline(Linear()))
residual_demand = t -> inter(1. + t / (24*3600) * samples) # 1. + is needed to avoid trying to access out of range

#########################################
#            SIM                     #
#########################################


vc1 = 1:N # ilc_nodes (here: without communication)
cover1 = Dict([v => [] for v in vc1])# ilc_cover
u = [zeros(1000,1);1;zeros(1000,1)];
fc = 1/6;
a = digitalfilter(Lowpass(fc),Butterworth(2));
Q1 = filtfilt(a,u);#Markov Parameter
Q = Toeplitz(Q1[1001:1001+n_updates_per_day-1],Q1[1001:1001+n_updates_per_day-1]);

# kappa_lst = (0:0.01:2) ./ l_hour
@everywhere begin
	update_lst_s= 60. *[30. : 30. : 240. ]
	kappa_lst_s = (0:.25:1.75) ./ update_lst_s
	kappa = kappa_lst_s[1]


	kappa_lst=repeat(kappa_lst_s,inner =length(update_lst_s))

	longueur=Int(length(kappa_lst)/length(update_lst_s))
	update_lst=repeat(update_lst_s,outer=longueur)

	num_monte = length(update_lst)*length(kappa_lst)
end

_compound_pars = system_structs.compound_pars(N, low_layer_control, kappa, vc1, cover1, Q, update)
_compound_pars.hl.daily_background_power .= 0
_compound_pars.hl.current_background_power .= 0
_compound_pars.hl.mismatch_yesterday .= 0.
_compound_pars.periodic_demand  = periodic_demand # t -> zeros(N) # periodic_demand
_compound_pars.residual_demand = residual_demand # t -> zeros(N) # residual_demand
_compound_pars.graph = graph_lst[1]
_compound_pars.coupling = 6 .* diagm(0=>ones(ne(graph_lst[1])))

@everywhere compound_pars = $_compound_pars


@everywhere begin
	factor = 0. # 0.01*rand(compound_pars.D * compound_pars.N) #0.001 #0.00001
	ic = factor .* ones(compound_pars.D * compound_pars.N)
	tspan = (0., num_days * l_day)
	ode_tl1 = ODEProblem(network_dynamics.ACtoymodel!, ic, tspan, compound_pars,
	callback=CallbackSet(PeriodicCallback(network_dynamics.Updating(),update ),
						 PeriodicCallback(network_dynamics.DailyUpdate_X, l_day)))
end


monte_prob = EnsembleProblem(
	ode_tl1,
	output_func = (sol, i) -> system_structs.observer_ic(sol, i, freq_filter, energy_filter, freq_threshold, num_days,N),
	prob_func = (prob,i,repeat) -> system_structs.prob_func_ic(prob, i, repeat, batch_size , kappa_lst, update_lst, num_days,kappa_lst_s,update_lst_s),

#	reduction = (u, data, I) -> system_structs.reduction_ic(u, data, I, batch_size),
	u_init = [])

res = solve(monte_prob,
					 Rodas4P(),
					 trajectories=num_monte,
					 batch_size=batch_size)


kappa = [p[6] for p in res.u]
hourly_energy = [p[10] for p in res.u]
norm_energy_d = [p[11] for p in res.u]

using LaTeXStrings
using Plots
using Vega
using Plotly


data = [
  [
    "z" => [2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51; 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52; 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53; 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54; 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55; 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56; 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57; 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58; 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59; 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60; 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61; 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62; 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63; 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64; 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65; 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66; 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67; 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68; 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69; 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70; 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71; 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72; 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73; 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74; 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75; 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76; 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77; 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78; 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79; 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80; 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81; 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82; 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83; 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84; 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85; 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86; 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87; 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88; 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89; 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90; 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91; 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92; 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93; 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94; 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95; 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96; 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97; 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98; 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99; 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100],
    "colorscale" => "Blackbody",
    "type" => "heatmap"
  ]
]
layout = ["title" => "Blackbody"]
response = Plotly.plot(data, ["layout" => layout, "filename" => "Blackbody-heatmap", "fileopt" => "overwrite"])
plot_url = response["url"]

plot(mean(norm_energy_d[5],dims=2),legend=:topright, label = "kappa = 1", ytickfontsize=14,
               xtickfontsize=14, linestyle =:solid, margin=8Plots.mm,left_margin=12Plots.mm,
    		   legendfontsize=8, linewidth=3,xaxis=("days [c]",font(14)), yaxis=("2-norm of the error",font(14)))  # ylims=(0,1e6)
plot!(mean(norm_energy_d[6],dims=2),label=  "kappa = 1.25 h^{-1}", linewidth = 3, linestyle=:dash)
plot!(mean(norm_energy_d[7],dims=2),label=  L"\kappa = 1.5\, h^{-1}", linewidth = 3, linestyle=:dashdot)
plot!(mean(norm_energy_d[8],dims=2),label=  L"\kappa = 1.75\, h^{-1}", linewidth = 3, linestyle=:dashdotdot)
savefig("$dir/20200319_kappa2_Y6_hetero.png")



using LaTeXStrings

plot(mean(norm_energy_d[5],dims=2),legend=:topright, label = L"\kappa = 1\, h^{-1}", ytickfontsize=14,
               xtickfontsize=14, linestyle =:solid, margin=8Plots.mm,left_margin=12Plots.mm,
    		   legendfontsize=8, linewidth=3,xaxis=("days [c]",font(14)), yaxis=("2-norm of the error",font(14)))  # ylims=(0,1e6)
plot!(mean(norm_energy_d[6],dims=2),label=  "kappa = 1.25 h^{-1}", linewidth = 3, linestyle=:dash)
plot!(mean(norm_energy_d[7],dims=2),label=  L"\kappa = 1.5\, h^{-1}", linewidth = 3, linestyle=:dashdot)
plot!(mean(norm_energy_d[8],dims=2),label=  L"\kappa = 1.75\, h^{-1}", linewidth = 3, linestyle=:dashdotdot)

#plot!(mean(norm_energy_d[9],dims=2), label= L"\kappa = 2 h^{-1}", linewidth = 3, linestyle=:dot)
#title!("Error norm")
savefig("$dir/20200319_kappa2_Y6_hetero.png")



# # never save the solutions INSIDE the git repo, they are too large, please make a folder solutions at the same level as the git repo and save them there
# jldopen("../../solutions/sol_def_N4.jld2", true, true, true, IOStream) do file
# 	file["sol1"] = sol1
# end
#
# @save "../../solutions/sol_kp525_ki0005_N4_pn_de-in_Q.jld2" sol1
