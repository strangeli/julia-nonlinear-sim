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
	#plotlyjs()
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

begin
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
	l_hour = 3600 # DemCurve.l_hour
	l_minute = 60 # DemCurve.l_minute
	#low_layer_control = experiments.LeakyIntegratorPars(M_inv=0.2,kP=52,T_inv=1/0.05,kI=10)
	#low_layer_control = experiments.LeakyIntegratorPars(M_inv=0.2,kP=525,T_inv=1/0.05,kI=0.005)
	#low_layer_control = experiments.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner=N),kI=repeat([0.005], inner=N)) # different for each node, change array
	low_layer_control = system_structs.LeakyIntegratorPars(M_inv=[1/5.; 1/4.8; 1/4.1; 1/4.8],kP= [400.; 110.; 100.; 200.],T_inv=[1/0.04; 1/0.045; 1/0.047; 1/0.043],kI=[0.05; 0.004; 0.05; 0.001]) # different for each node, change array
	#low_layer_control = experiments.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=[0.1; 10; 100; 1000],T_inv=repeat([1/0.05], inner=N),kI=repeat([0.005], inner=N)) # different for each node, change array
	#low_layer_control = experiments.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=[1/0.05; 1/0.5; 1/5; 1/50],kI=repeat([0.005], inner=N)) # different for each node, change array
	#low_layer_control = experiments.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner = N),kI=[0.005; 0.5; 5; 500]) # different for each node, change array
	#low_layer_control = experiments.LeakyIntegratorPars(M_inv=[0.002; 0.2; 2; 20],kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner = N),kI=repeat([0.005], inner=N)) # different for each node, change array
	kappa = 0.75 / l_hour
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
Q = Toeplitz(Q1[1001:1001+24-1],Q1[1001:1001+24-1]);

# kappa_lst = (0:0.01:2) ./ l_hour
@everywhere begin
	kappa_lst = (0:.25:2.) ./ l_hour
	kappa = kappa_lst[1]
	num_monte = batch_size*length(kappa_lst)
end

_compound_pars = system_structs.compound_pars(N, low_layer_control, kappa, vc1, cover1, Q, 1)
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
	callback=CallbackSet(PeriodicCallback(network_dynamics.HourlyUpdate(), l_hour),
						 PeriodicCallback(network_dynamics.DailyUpdate_X, l_day)))
end


monte_prob = EnsembleProblem(
	ode_tl1,
	output_func = (sol, i) ->system_structs.observer_ic(sol, i, freq_filter, energy_filter, freq_threshold, num_days,N),
	prob_func = (prob,i,repeat) -> system_structs.prob_func_ic(prob,i,repeat, batch_size, kappa_lst, num_days),
#	reduction = (u, data, I) -> experiments.reduction_ic(u, data, I, batch_size),
	u_init = [])

res = solve(monte_prob,
					 Rodas4P(),
					 trajectories=num_monte,
					 batch_size=batch_size)

kappa = [p[6] for p in res.u]
hourly_energy = [p[10] for p in res.u]
norm_energy_d = [p[11] for p in res.u]
norm_energy_dn = [p[12] for p in res.u]

using LaTeXStrings
plot(mean(norm_energy_d[1],dims=2),legend=:right, label = L"\kappa = 0.00\, h^{-1}", ytickfontsize=14,
               xtickfontsize=14, linestyle=:dot, margin=8Plots.mm,
    		   legendfontsize=14, linewidth=3,xaxis=("days [c]",font(14)), yaxis = ("2-norm of the error [s]",font(14)), left_margin=12Plots.mm) #  ylims=(0,1e6)
plot!(mean(norm_energy_d[2],dims=2), label= L"\kappa = 0.25\, h^{-1}", linewidth = 3, linestyle=:dashdotdot)
plot!(mean(norm_energy_d[3],dims=2), label= L"\kappa = 0.50\, h^{-1}", linewidth = 3, linestyle=:dashdot)
plot!(mean(norm_energy_d[4],dims=2),label=  L"\kappa = 0.75\, h^{-1}", linewidth = 3, linestyle=:dash)
plot!(mean(norm_energy_d[5],dims=2), label= L"\kappa = 1.00\, h^{-1}", linewidth = 3, linestyle=:solid)
#title!("Error norm")
savefig("$dir/201207_kappa_Y6_hetero.png")

using LaTeXStrings
plot(mean(norm_energy_d[5],dims=2),legend=:right, label = L"\kappa = 1.00\, h^{-1}", ytickfontsize=14,
               xtickfontsize=14, linestyle =:solid, margin=8Plots.mm,left_margin=12Plots.mm,
    		   legendfontsize=13, linewidth=3,xaxis=("days [c]",font(14)), yaxis=("2-norm of the error [s]",font(14)))  # ylims=(0,1e6)
plot!(mean(norm_energy_d[6],dims=2),label=  L"\kappa = 1.25\, h^{-1}", linewidth = 3, linestyle=:dash)
plot!(mean(norm_energy_d[7],dims=2),label=  L"\kappa = 1.50\, h^{-1}", linewidth = 3, linestyle=:dashdot)
plot!(mean(norm_energy_d[8],dims=2),label=  L"\kappa = 1.75\, h^{-1}", linewidth = 3, linestyle=:dashdotdot)
plot!(mean(norm_energy_d[9],dims=2), label= L"\kappa = 2.00\, h^{-1}", linewidth = 3, linestyle=:dot)
#title!("Error norm")
savefig("$dir/201207_kappa2_Y6_hetero.png")

# using LaTeXStrings
# plot(norm_energy_dn[1],legend=:right, label = L"\kappa = 0.00\, h^{-1}", ytickfontsize=14,
#                xtickfontsize=14, linestyle=:dot, margin=8Plots.mm,
#     		   legendfontsize=14, linewidth=3,xaxis=("days [c]",font(14)), yaxis = ("2-norm of the error [s]",font(14)),left_margin=12Plots.mm) #  ylims=(0,1e6)
# plot!(norm_energy_dn[2], label= L"\kappa = 0.25\, h^{-1}", linewidth = 3, linestyle=:dashdotdot)
# plot!(norm_energy_dn[3], label= L"\kappa = 0.50\, h^{-1}", linewidth = 3, linestyle=:dashdot)
# plot!(norm_energy_dn[4],label=  L"\kappa = 0.75\, h^{-1}", linewidth = 3, linestyle=:dash)
# plot!(norm_energy_dn[5], label= L"\kappa = 1.00\, h^{-1}", linewidth = 3, linestyle=:solid)
# #title!("Error norm")
# savefig("$dir/201207_kappa_Y6_hetero_dn.png")

# using LaTeXStrings
# plot(norm_energy_dn[5],legend=:right, label = L"\kappa = 1.00\, h^{-1}", ytickfontsize=14,
#                xtickfontsize=14, linestyle =:solid, margin=8Plots.mm,left_margin=12Plots.mm,
#     		   legendfontsize=13, linewidth=3,xaxis=("days [c]",font(14)), yaxis=("2-norm of the error [s]",font(14)), linecolor = :black)  # ylims=(0,1e6)
# plot!(norm_energy_dn[6],label=  L"\kappa = 1.25\, h^{-1}", linewidth = 3, linestyle=:dash)
# plot!(norm_energy_dn[7],label=  L"\kappa = 1.50\, h^{-1}", linewidth = 3, linestyle=:dashdot)
# plot!(norm_energy_dn[8],label=  L"\kappa = 1.75\, h^{-1}", linewidth = 3, linestyle=:dashdotdot)
# plot!(norm_energy_dn[9], label= L"\kappa = 2.00\, h^{-1}", linewidth = 3, linestyle=:dot)
# #title!("Error norm")
# savefig("$dir/201207_kappa2_Y6_hetero_dn.png")

# # never save the solutions INSIDE the git repo, they are too large, please make a folder solutions at the same level as the git repo and save them there
# jldopen("../../solutions/sol_def_N4.jld2", true, true, true, IOStream) do file
# 	file["sol1"] = sol1
# end
#
# @save "../../solutions/sol_kp525_ki0005_N4_pn_de-in_Q.jld2" sol1
