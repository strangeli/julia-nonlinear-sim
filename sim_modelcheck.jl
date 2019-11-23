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
	num_days =  20
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
	low_layer_control = experiments.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner=N),kI=repeat([0.005], inner=N)) # different for each node, change array
	#low_layer_control = experiments.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=[0.1; 10; 100; 1000],T_inv=repeat([1/0.05], inner=N),kI=repeat([0.005], inner=N)) # different for each node, change array
	#low_layer_control = experiments.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=[1/0.05; 1/0.5; 1/5; 1/50],kI=repeat([0.005], inner=N)) # different for each node, change array
	#low_layer_control = experiments.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner = N),kI=[0.005; 0.5; 5; 500]) # different for each node, change array
	#low_layer_control = experiments.LeakyIntegratorPars(M_inv=[0.002; 0.2; 2; 20],kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner = N),kI=repeat([0.005], inner=N)) # different for each node, change array
	kappa = 1.0 / l_hour
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

struct random_time_series
	rand_ts
end

function (rts::random_time_series)(t,delta)
	index = Int(floor(t / (delta)))
	rts.rand_ts[index + 1,:]
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
#demand_ampp = demand_amp_var(repeat([10 10 10 10 10 80 80 80 80 80 80 10 10 10 10 10 40 40 40 40 40], outer=Int(N/2))') # random positive amp over days by 10%
#demand_ampn = demand_amp_var(repeat([10 10 10 10 10 80 80 80 80 80 80 10 10 10 10 10 40 40 40 40 40], outer=Int(N/2))') # random positive amp over days by 10%
#demand_amp = t->vcat(demand_ampp(t), demand_ampn(t))


# # random positive amp over days by 30%
# demand_ampp = demand_amp_var(70 .+ rand(num_days+1,Int(N/2)).* 30.)
# demand_ampn = demand_amp_var(70 .+ rand(num_days+1,Int(N/2)).* 30.)  # random negative amp over days by 10%
# demand_amp = t->vcat(demand_ampp(t), demand_ampn(t))

periodic_demand =  t-> demand_amp(t) .* sin(t*pi/(24*3600))^2
samples = 24*4
inter = interpolate([.2 * randn(N) for i in 1:(num_days * samples + 1)], BSpline(Linear()))
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


_compound_pars = experiments.compound_pars(N, low_layer_control, kappa, vc1, cover1, Q)


R = random_time_series(1e-5.*rand(24*num_days*l_hour*10,N))

_compound_pars.hl.daily_background_power .= 0. #ones(24,N) #t -> (ones(24,N)+reshape(R(t,0.1),N,24))'#ones(24,N)#
_compound_pars.hl.current_background_power = t->ones(N)  #t -> (ones(N)+R(t,0.1))  # ones(N)
_compound_pars.hl.mismatch_yesterday .= 0. #[my;-my]'
_compound_pars.periodic_demand  = t -> zeros(N) # t -> zeros(N) #periodic_demand
_compound_pars.residual_demand = t -> zeros(N) #t -> zeros(N) #residual_demand
_compound_pars.graph = graph_lst[1]

@everywhere compound_pars = $_compound_pars
#factor = 0.01*rand(compound_pars.D * compound_pars.N)

@everywhere begin
	#0.001#0.00001
	ic = factor .* ones(compound_pars.D * compound_pars.N)
	tspan = (0., num_days *l_day/10)
#	ode_tl1 = ODEProblem(network_dynamics.ACtoymodel!, ic, tspan, compound_pars,
#	callback=CallbackSet(PeriodicCallback(network_dynamics.HourlyUpdate(), l_hour),
#						 PeriodicCallback(network_dynamics.DailyUpdate_X, l_day)))
	ode_tl0_lin = ODEProblem(network_dynamics.ACtoymodel_lin!, ic, tspan, compound_pars)
    ode_tl0 = ODEProblem(network_dynamics.ACtoymodel!, ic, tspan, compound_pars)
end


#sol1 = solve(ode_tl1, Rodas4())
#sol0_lin = solve(ode_tl0_lin, Rodas4())
sol0 = solve(ode_tl0, Rodas4())
#CSV.write("$dir/files/20191021_sol0_modelcheck_lin.csv",  DataFrame(sol0_lin'), writeheader=false)
CSV.write("$dir/files/20191022_sol0_modelcheck_P.csv",  DataFrame(sol0'), writeheader=false)

#Rr = t -> ones(24,N)+reshape(R(t,0.1),N,24)'
#plot(0:0.001:10,t -> Rr(t)[1])
#savefig("$dir/plots/modelcheck_u_dist.png")

hourly_energy_d = zeros(Int(24*num_days/10),N)
#sol1 = sol0
for i=1:Int(24*num_days/10)
	hourly_energy_d[i,1] = sol1(i*3600)[energy_filter[1]]
	hourly_energy_d[i,2] = sol1(i*3600)[energy_filter[2]]
	hourly_energy_d[i,3] = sol1(i*3600)[energy_filter[3]]
	hourly_energy_d[i,4] = sol1(i*3600)[energy_filter[4]]
end
plot(hourly_energy_d, label = "y2")
savefig("$dir/plots/modelcheck_P_dist.png")

#sol0_u1 = CSV.read("$dir/files/20191022_sol0_modelcheck_P.csv", header=false)
#sol1 = sol0_u1
hourly_energy = zeros(Int(24*num_days/10),N)
for i=1:Int(24*num_days/10)
	hourly_energy[i,1] = sol0(i*3600)[energy_filter[1]]
	hourly_energy[i,2] = sol0(i*3600)[energy_filter[2]]
	hourly_energy[i,3] = sol0(i*3600)[energy_filter[3]]
	hourly_energy[i,4] = sol0(i*3600)[energy_filter[4]]
end
plot!(hourly_energy, label= "y1")
savefig("$dir/plots/modelcheck_P.png")

using MAT
file = matopen("$dir/../Matlab_matrices/data/ABCP_N4_Yfull.mat")
P = read(file, "P")  #note that this does NOT introduce a variable ``varname`` into scope
close(file)

EPS = zeros(24*N)
for i = 1:24
	EPS[(i-1)*4+1:i*4] = R(i*3600,0.1)
end

y = reshape(P*EPS,4,24)'

#Puf = P*EPS

plot(hourly_energy_d-hourly_energy, label = "y2-y1")
plot!(1:48,[y;y], label = "P*delta")
#plot!(1:480,t->Puf(t)[2], label = "P*delta, node 2")
#plot!(1:480,t->Puf(t)[3], label = "P*delta, node 3")
#plot!(1:480,t->Puf(t)[4], label = "P*delta, node 4")

plot(sol0_lin,vars=phase_filter,tspan=(0,200),label="linear")
plot!(sol0,vars=phase_filter,tspan=(0,200), label="nonlinear")
ylabel!("phase [rad]")
xlabel!("time [s]")
savefig("$dir/plots/modelcheck_phase.png")

plot(sol0_lin,vars=freq_filter,tspan=(0,200),label="linear")
plot!(sol0,vars=freq_filter,tspan=(0,200), label="nonlinear")
ylabel!("freq [rad/s]")
xlabel!("time [s]")
savefig("$dir/plots/modelcheck_freq.png")

plot(sol0_lin,vars=control_filter,tspan=(0,200),label="linear")
plot!(sol0,vars=control_filter,tspan=(0,200), label="nonlinear")
ylabel!("chi [Ws]")
xlabel!("time [s]")
savefig("$dir/plots/modelcheck_chi.png")

plot(sol0_lin,vars=energy_filter,tspan=(0,200),label="linear")
plot!(sol0,vars=energy_filter,tspan=(0,200), label="nonlinear")


hourly_energy = zeros(24*num_days,N)
for i=1:24*num_days
	hourly_energy[i,1] = sol1(i*3600)[energy_filter[1]]
	hourly_energy[i,2] = sol1(i*3600)[energy_filter[2]]
	hourly_energy[i,3] = sol1(i*3600)[energy_filter[3]]
	hourly_energy[i,4] = sol1(i*3600)[energy_filter[4]]
end
plot(hourly_energy)

ILC_power = zeros(num_days,24,N)
for i=2:num_days
	ILC_power[i,:,1] = Q*(ILC_power[i-1,:,1] +  kappa*hourly_energy[(i-1)*24+1:i*24,1])
	ILC_power[i,:,2] = Q*(ILC_power[i-1,:,2] +  kappa*hourly_energy[(i-1)*24+1:i*24,2])
	ILC_power[i,:,3] = Q*(ILC_power[i-1,:,3] +  kappa*hourly_energy[(i-1)*24+1:i*24,3])
	ILC_power[i,:,4] = Q*(ILC_power[i-1,:,4] +  kappa*hourly_energy[(i-1)*24+1:i*24,4])
end

ILC_power_agg = maximum(mean(ILC_power.^2,dims=3),dims=2)

load_amp = [first(demand_ampp(t)) for t in 1:3600*24:3600*24*num_days]
using LaTeXStrings


plot(1:24:num_days*24, ILC_power_agg[:,1,1] ./ maximum(ILC_power_agg), label=L"$\max_h \Vert P_{ILC, k}\Vert$", legend=:topleft)
plot!(1:24*num_days,mean(hourly_energy, dims=2) ./ maximum(hourly_energy), label=L"y_h")
plot!(1:24:24*num_days, load_amp  ./ maximum(load_amp), label = "demand amplitude")
xlabel!("hour h [h]")
ylabel!("normed quantities [a.u.]")
savefig("$dir/plots/yh_demand_ILC_.png")

#sol2 = CSV.read("$dir/files/test2.csv", header=false)
#plot(sol2[:Column1])

#p1 = plot()
#plot(sol2[Symbol("0.0_9")])

# monte_prob = EnsembleProblem(
# 	ode_tl1,
# 	output_func = (sol, i) -> experiments.observer_ic(sol, i, freq_filter, energy_filter, freq_threshold),
# 	prob_func = (prob,i,repeat) -> experiments.prob_func_ic(prob,i,repeat, batch_size, graph_lst, num_days),
#	reduction = (u, data, I) -> experiments.reduction_ic(u, data, I, batch_size),
#	u_init = [])
#
# res = @time solve(monte_prob,
# 					 Rodas4P(),
# 					 trajectories=num_monte,
# 					 batch_size=batch_size)




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
