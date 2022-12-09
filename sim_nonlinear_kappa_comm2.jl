	using FileIO, GraphIO, CSV, DataFrames
	using Distributed
	using Interpolations

	_calc = true
	_slurm = true

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
		calc = $_calc	 # if false, only plotting
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
		using LaTeXStrings
		using Plots
		Random.seed!(42)
	end

	@everywhere begin
		N = 24
		num_days =  20
		batch_size = 1
	end

	@everywhere begin
		freq_threshold = 0.001
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
		low_layer_control = system_structs.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner=N),kI=repeat([0.005], inner=N)) # different for each node, change array
		#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=[1/5.; 1/4.8; 1/4.1; 1/4.8],kP= [400.; 110.; 100.; 200.],T_inv=[1/0.04; 1/0.045; 1/0.047; 1/0.043],kI=[0.05; 0.004; 0.05; 0.001]) # different for each node, change array
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

	@everywhere struct demand_amp_var
		demand
	end

	@everywhere function (dav::demand_amp_var)(t)
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


	# u = [zeros(1000,1);1;zeros(1000,1)];
	# fc = 1/6;
	# a = digitalfilter(Lowpass(fc),Butterworth(2));
	# Q1 = filtfilt(a,u);#Markov Parameter
	# Q = Toeplitz(Q1[1001:1001+24-1],Q1[1001:1001+24-1]);
 	Q = diagm(0=>ones(24))

	# kappa_lst = (0:0.01:2) ./ l_hour
	@everywhere begin
		#kappa_lst = (0:.25:1.75) ./ l_hour
		kappa_lst = (0:.1:.7) ./ l_hour
		kappa = kappa_lst[1]
		num_monte = batch_size*length(kappa_lst)
		lambda_lst = 1.
		lambda = 1.
		obs_days = num_days
	end

	#############################################################################################
	#############################################################################################
	############################################################################################


	vc1 = 1:N # ilc_nodes (here: without communication)
	cover1 = Dict([v => [] for v in vc1])# ilc_cover

	_compound_pars = system_structs.compound_pars(N, low_layer_control, kappa, vc1, cover1, Q, lambda)
	_compound_pars.hl.daily_background_power .= 0
	_compound_pars.hl.current_background_power .= 0
	_compound_pars.hl.mismatch_yesterday .= 0.
	_compound_pars.periodic_demand  = periodic_demand # t -> zeros(N) # periodic_demand
	_compound_pars.residual_demand = residual_demand # t -> zeros(N) # residual_demand
	_compound_pars.graph = graph_lst[1]
	_compound_pars.coupling = 6 .* diagm(0=>ones(ne(graph_lst[1])))

	@everywhere compound_pars = $_compound_pars

begin
	exp_case = "I"

	@everywhere begin
	factor = 0. # 0.01*rand(compound_pars.D * compound_pars.N) #0.001 #0.00001
	ic = factor .* ones(compound_pars.D * compound_pars.N)
	tspan = (0., num_days * l_day)
	ode_I = ODEProblem(network_dynamics.ACtoymodel!, ic, tspan, compound_pars,
	callback=CallbackSet(PeriodicCallback(network_dynamics.HourlyUpdateEcon(), l_hour),
					 PeriodicCallback(network_dynamics.DailyUpdate, l_day)))
	end


	monte_prob_I = EnsembleProblem(
	ode_I,
	output_func = (sol, i) -> system_structs.observer_basic_types(sol, i, energy_filter), # what should be extracted from one run
	prob_func = (prob,i,repeat) -> system_structs.prob_func_I(prob, i, repeat, batch_size, num_days, kappa_lst),
	#	reduction = (u, data, I) -> experiments.reduction_ic(u, data, I, batch_size),
	u_init = [])

	res_I = solve(monte_prob_I,
				 Rodas4P(),
				 trajectories=num_monte,
				 batch_size=batch_size, EnsembleDistributed())


	##############################################
	######### comm #######################
	##############################################

	kappa_I = [p[1] for p in res_I.u] # p[6]
	hourly_energy_I = [p[2] for p in res_I.u] # p[10]
	norm_energy_d_I = [p[3] for p in res_I.u] # p[11]

	using Dates
	date = Dates.now()
	CSV.write("$dir/data/$(date)_error_I_sign.csv", DataFrame([[mean(norm_energy_d_I[i],dims=2) for i in 1:8][j][k] for j in 1:8, k in 1:20]'))

	#norm_energy_d_I = CSV.read("$dir/2020-10-09T23:37:05.044_error_I_sign.csv")

	#######################################################

	using LaTeXStrings
	plot(mean(norm_energy_d_I[1],dims=2),legend=:topright, label = L"\kappa = 0\, h^{-1}", ytickfontsize=14,
	           xtickfontsize=14, linestyle=:dot, margin=8Plots.mm, left_margin=15Plots.mm, ylims =(0,9900),
			   legendfontsize=14, linewidth=3,xaxis=("days [c]",font(14)), yaxis = ("2-norm of the error",font(14))) #  ylims=(0,1e6)
	plot!(mean(norm_energy_d_I[2],dims=2), label= L"\kappa = 0.1\, h^{-1}", linewidth = 3, linestyle=:dashdotdot)
	plot!(mean(norm_energy_d_I[3],dims=2), label= L"\kappa = 0.2\, h^{-1}", linewidth = 3, linestyle=:dashdot)
	plot!(mean(norm_energy_d_I[4],dims=2),label=  L"\kappa = 0.3\, h^{-1}", linewidth = 3, linestyle=:dash)
	plot!(mean(norm_energy_d_I[5],dims=2), label= L"\kappa = 0.4\, h^{-1}", linewidth = 3, linestyle=:solid)
	title!("Error norm for scenario I")
	savefig("$dir/plots/$(date)_kappa_Y6_homo_N_$(N)_I_sign.png")

	using LaTeXStrings
	plot(mean(norm_energy_d_I[5],dims=2),legend=:topright, label = L"\kappa = 0.4\, h^{-1}", ytickfontsize=14,
	           xtickfontsize=14, linestyle =:solid, margin=8Plots.mm,left_margin=15Plots.mm, ylims =(0,9900),
			   legendfontsize=13, linewidth=3,xaxis=("days [c]",font(14)), yaxis=("2-norm of the error",font(14)))  # ylims=(0,1e6)
	plot!(mean(norm_energy_d_I[6],dims=2),label=  L"\kappa = 0.5\, h^{-1}", linewidth = 3, linestyle=:dash)
	plot!(mean(norm_energy_d_I[7],dims=2),label=  L"\kappa = 0.6\, h^{-1}", linewidth = 3, linestyle=:dashdot)
	plot!(mean(norm_energy_d_I[8],dims=2),label=  L"\kappa = 0.7\, h^{-1}", linewidth = 3, linestyle=:dashdotdot)
	#plot!(mean(norm_energy_d[9],dims=2), label= L"\kappa = 2 h^{-1}", linewidth = 3, linestyle=:dot)
	title!("Error norm for scenario I")
	savefig("$dir/plots/$(date)_kappa2_Y6_homo_N_$(N)_I_sign.png")
end
	##############################################################################################
	###############################################################################################
	##############################################################################################

begin
	exp_case = "II"

	vc2 = independent_set(graph_lst[1], DegreeIndependentSet()) # ilc_nodes
	cover2 = Dict([v => neighbors(graph_lst[1], v) for v in vc2]) # ilc_cover
	_compound_pars = system_structs.compound_pars(N, low_layer_control, kappa, vc2, cover2, Q, lambda)
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
		ode_II = ODEProblem(network_dynamics.ACtoymodel!, ic, tspan, compound_pars,
		callback=CallbackSet(PeriodicCallback(network_dynamics.HourlyUpdateEcon(), l_hour),
						 PeriodicCallback(network_dynamics.DailyUpdate_II, l_day)))
	end


	monte_prob_II = EnsembleProblem(
	ode_II,
	output_func = (sol, i) -> system_structs.observer_basic_types(sol, i,energy_filter), # what should be extracted from one run
	prob_func = (prob,i,repeat) -> system_structs.prob_func_II(prob, i, repeat, batch_size, num_days, kappa_lst),
	#	reduction = (u, data, I) -> experiments.reduction_ic(u, data, I, batch_size),
	u_init = [])

	res_II = solve(monte_prob_II,
				 Rodas4P(),
				 trajectories=num_monte,
				 batch_size=batch_size, EnsembleDistributed())


	##############################################
	######### comm #######################
	##############################################

	kappa_II = [p[1] for p in res_II.u] # p[6]
	hourly_energy_II = [p[2] for p in res_II.u] # p[10]
	norm_energy_d_II = [p[3] for p in res_II.u] # p[11]

	using Dates
	date = Dates.now()
	CSV.write("$dir/data/$(date)_error_II_sign.csv", DataFrame([[mean(norm_energy_d_II[i],dims=2) for i in 1:8][j][k] for j in 1:8, k in 1:20]'))
	# res = CSV.read("$dir/error.csv.csv")

	#######################################################

	using LaTeXStrings
	plot(mean(norm_energy_d_II[1],dims=2),legend=:topright, label = L"\kappa = 0\, h^{-1}", ytickfontsize=14,
	           xtickfontsize=14, linestyle=:dot, margin=8Plots.mm, left_margin=15Plots.mm, ylims =(0,9900),
			   legendfontsize=14, linewidth=3,xaxis=("days [c]",font(14)), yaxis = ("2-norm of the error",font(14))) #  ylims=(0,1e6)
	plot!(mean(norm_energy_d_II[2],dims=2), label= L"\kappa = 0.1\, h^{-1}", linewidth = 3, linestyle=:dashdotdot)
	plot!(mean(norm_energy_d_II[3],dims=2), label= L"\kappa = 0.2\, h^{-1}", linewidth = 3, linestyle=:dashdot)
	plot!(mean(norm_energy_d_II[4],dims=2),label=  L"\kappa = 0.3\, h^{-1}", linewidth = 3, linestyle=:dash)
	plot!(mean(norm_energy_d_II[5],dims=2), label= L"\kappa = 0.4\, h^{-1}", linewidth = 3, linestyle=:solid)
	title!("Error norm for scenario II")
	savefig("$dir/plots/$(date)_kappa_Y6_homo_N_$(N)_II_sign.png")

	using LaTeXStrings
	plot(mean(norm_energy_d_II[5],dims=2),legend=:topright, label = L"\kappa = 0.4\, h^{-1}", ytickfontsize=14,
	           xtickfontsize=14, linestyle =:solid, margin=8Plots.mm, left_margin=15Plots.mm, ylims =(0,9900),
			   legendfontsize=14, linewidth=3, xaxis=("days [c]",font(14)), yaxis=("2-norm of the error",font(14)))  # ylims=(0,1e6)
	plot!(mean(norm_energy_d_II[6],dims=2),label=  L"\kappa = 0.5\, h^{-1}", linewidth = 3, linestyle=:dash)
	plot!(mean(norm_energy_d_II[7],dims=2),label=  L"\kappa = 0.6\, h^{-1}", linewidth = 3, linestyle=:dashdot)
	plot!(mean(norm_energy_d_II[8],dims=2),label=  L"\kappa = 0.7\, h^{-1}", linewidth = 3, linestyle=:dashdotdot)
	#plot!(mean(norm_energy_d[9],dims=2), label= L"\kappa = 2 h^{-1}", linewidth = 3, linestyle=:dot)
	title!("Error norm for scenario II")
	savefig("$dir/plots/$(date)_kappa2_Y6_homo_N_$(N)_II_sign.png")
end

	##############################################################################################
	###############################################################################################
	##############################################################################################
begin
		exp_case = "III"

		vc3 = vc2 # ilc_nodes
		cover3 = Dict([v => [] for v in vc3]) # ilc_cover
	   _compound_pars = system_structs.compound_pars(N, low_layer_control, kappa, vc3, cover3, Q, lambda)
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
			ode_III = ODEProblem(network_dynamics.ACtoymodel!, ic, tspan, compound_pars,
			callback=CallbackSet(PeriodicCallback(network_dynamics.HourlyUpdateEcon(), l_hour),
								 PeriodicCallback(network_dynamics.DailyUpdate_III, l_day)))
		end


	monte_prob_III = EnsembleProblem(
		ode_III,
		output_func = (sol, i) -> system_structs.observer_basic_types(sol, i, energy_filter), # what should be extracted from one run
		prob_func = (prob,i,repeat) -> system_structs.prob_func_III(prob, i, repeat, batch_size, num_days, kappa_lst),
	#	reduction = (u, data, I) -> experiments.reduction_ic(u, data, I, batch_size),
		u_init = [])

	res_III = solve(monte_prob_III,
						 Rodas4P(),
						 trajectories=num_monte,
						 batch_size=batch_size, EnsembleDistributed())


	##############################################
	######### comm #######################
	##############################################

	kappa_III = [p[1] for p in res_III.u] # p[6]
	hourly_energy_III = [p[2] for p in res_III.u] # p[10]
	norm_energy_d_III = [p[3] for p in res_III.u] # p[11]

	using Dates
	date = Dates.now()
	CSV.write("$dir/data/$(date)_error_III_sign.csv", DataFrame([[mean(norm_energy_d_III[i],dims=2) for i in 1:8][j][k] for j in 1:8, k in 1:20]'))
	# res = CSV.read("$dir/error.csv.csv")

	#######################################################

		using LaTeXStrings
		plot(mean(norm_energy_d_III[1],dims=2),legend=:topright, label = L"\kappa = 0\, h^{-1}", ytickfontsize=14,
		               xtickfontsize=14, linestyle=:dot, margin=8Plots.mm, left_margin=15Plots.mm, ylims =(0,9900),
		    		   legendfontsize=14, linewidth=3,xaxis=("days [c]",font(14)), yaxis = ("2-norm of the error",font(14))) #  ylims=(0,1e6)
		plot!(mean(norm_energy_d_III[2],dims=2), label= L"\kappa = 0.1\, h^{-1}", linewidth = 3, linestyle=:dashdotdot)
		plot!(mean(norm_energy_d_III[3],dims=2), label= L"\kappa = 0.2\, h^{-1}", linewidth = 3, linestyle=:dashdot)
		plot!(mean(norm_energy_d_III[4],dims=2),label=  L"\kappa = 0.3\, h^{-1}", linewidth = 3, linestyle=:dash)
		plot!(mean(norm_energy_d_III[5],dims=2), label= L"\kappa = 0.4\, h^{-1}", linewidth = 3, linestyle=:solid)
		title!("Error norm for scenario III")
		savefig("$dir/plots/$(date)_kappa_Y6_homo_N_$(N)_III_sign.png")

		using LaTeXStrings
		plot(mean(norm_energy_d_III[5],dims=2),legend=:topright, label = L"\kappa = 0.4\, h^{-1}", ytickfontsize=14,
		               xtickfontsize=14, linestyle =:solid, margin=8Plots.mm, left_margin=15Plots.mm, ylims =(0,9900),
		    		   legendfontsize=13, linewidth=3,xaxis=("days [c]",font(14)), yaxis=("2-norm of the error",font(14)))  # ylims=(0,1e6)
		plot!(mean(norm_energy_d_III[6],dims=2),label=  L"\kappa = 0.5\, h^{-1}", linewidth = 3, linestyle=:dash)
		plot!(mean(norm_energy_d_III[7],dims=2),label=  L"\kappa = 0.6\, h^{-1}", linewidth = 3, linestyle=:dashdot)
		plot!(mean(norm_energy_d_III[8],dims=2),label=  L"\kappa = 0.7\, h^{-1}", linewidth = 3, linestyle=:dashdotdot)
		#plot!(mean(norm_energy_d[9],dims=2), label= L"\kappa = 2 h^{-1}", linewidth = 3, linestyle=:dot)
		title!("Error norm for scenario III")
		savefig("$dir/plots/$(date)_kappa2_Y6_homo_N_$(N)_III_sign.png")
end


		##############################################################################################
		###############################################################################################
		##############################################################################################
begin
		exp_case = "IV"

		if exp_case == "IV"
			ilc_ratio = Int(N/2)
			vc4 = sample(1:N,ilc_ratio,replace = false) # ilc_nodes
			ilcm4 =  diagm(0 => zeros(N))
			cover4 = [] # ilc_covers
			for i in 1:ilc_ratio
				ilcm4[vc4[i],vc4[i]] = 1
				#kappa[few[1], vc[i]] = 1 # here multiple entries
				#kappa[few[2], vc[i]] = 1
				#kappa[few[3], vc[i]] = 1
				a = 1:vc4[i]-1
				b = vc4[i]+1:N
				c = [collect(a); collect(b)]
				few = sample(c,3,replace = false)
				ilcm4[vc4[i], few[1]] = 1
				ilcm4[vc4[i], few[2]] = 1
				ilcm4[vc4[i], few[3]] = 1
				push!(cover4, Dict([vc4[i] => few]))
		end
				ilcm4 .*= 0.25
				#kappa .*=  ilcm4
				kappa_lst_IV = [kappa_lst[i] .* ilcm4 for i in 1:length(kappa_lst)]
			end

			_compound_pars = system_structs.compound_pars(N, low_layer_control, kappa_lst_IV[1], vc4, cover4, Q, lambda)
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
				ode_IV = ODEProblem(network_dynamics.ACtoymodel!, ic, tspan, compound_pars,
				callback=CallbackSet(PeriodicCallback(network_dynamics.HourlyUpdateEcon(), l_hour),
									 PeriodicCallback(network_dynamics.DailyUpdate_IV, l_day)))
			end


			monte_prob_IV = EnsembleProblem(
				ode_IV,
				output_func = (sol, i) -> system_structs.observer_basic_types(sol, i, energy_filter), # what should be extracted from one run
				prob_func = (prob,i,repeat) -> system_structs.prob_func_IV(prob, i, repeat, batch_size, num_days, kappa_lst_IV),
			#	reduction = (u, data, I) -> experiments.reduction_ic(u, data, I, batch_size),
				u_init = [])

			res_IV = solve(monte_prob_IV,
								 Rodas4P(),
								 trajectories=num_monte,
								 batch_size=batch_size, EnsembleDistributed())


			##############################################
			######### comm #######################
			##############################################

			kappa_IV = [p[1] for p in res_IV.u] # p[6]
			hourly_energy_IV = [p[2] for p in res_IV.u] # p[10]
			norm_energy_d_IV = [p[3] for p in res_IV.u] # p[11]

			using Dates
			date = Dates.now()
			CSV.write("$dir/data/$(date)_error_IV_sign.csv", DataFrame([[mean(norm_energy_d_IV[i],dims=2) for i in 1:8][j][k] for j in 1:8, k in 1:20]'))
			# res = CSV.read("$dir/error.csv.csv")

		#######################################################

				using LaTeXStrings
				plot(mean(norm_energy_d_IV[1],dims=2),legend=:topright, label = L"\kappa = 0\, h^{-1}", ytickfontsize=14,
				               xtickfontsize=14, linestyle=:dot, margin=8Plots.mm, left_margin=15Plots.mm, ylims =(0,9900),
				    		   legendfontsize=14, linewidth=3,xaxis=("days [c]",font(14)), yaxis = ("2-norm of the error", font(14))) #  ylims=(0,1e6)
				plot!(mean(norm_energy_d_IV[2],dims=2), label= L"\kappa = 0.1\, h^{-1}", linewidth = 3, linestyle=:dashdotdot)
				plot!(mean(norm_energy_d_IV[3],dims=2), label= L"\kappa = 0.2\, h^{-1}", linewidth = 3, linestyle=:dashdot)
				plot!(mean(norm_energy_d_IV[4],dims=2),label=  L"\kappa = 0.3\, h^{-1}", linewidth = 3, linestyle=:dash)
				plot!(mean(norm_energy_d_IV[5],dims=2), label= L"\kappa = 0.4\, h^{-1}", linewidth = 3, linestyle=:solid)
				title!("Error norm for scenario IV")
				savefig("$dir/plots/$(date)_kappa_Y6_homo_N_$(N)_IV_sign.png")

				using LaTeXStrings
				plot(mean(norm_energy_d_IV[5],dims=2),legend=:topright, label = L"\kappa = 0.4\, h^{-1}", ytickfontsize=14,
				               xtickfontsize=14, linestyle =:solid, margin=8Plots.mm, left_margin=15Plots.mm, ylims =(0,9900),
				    		   legendfontsize=13, linewidth=3,xaxis=("days [c]",font(14)), yaxis=("2-norm of the error",font(14)))  # ylims=(0,1e6)
				plot!(mean(norm_energy_d_IV[6],dims=2),label=  L"\kappa = 0.5\, h^{-1}", linewidth = 3, linestyle=:dash)
				plot!(mean(norm_energy_d_IV[7],dims=2),label=  L"\kappa = 0.6\, h^{-1}", linewidth = 3, linestyle=:dashdot)
				plot!(mean(norm_energy_d_IV[8],dims=2),label=  L"\kappa = 0.7\, h^{-1}", linewidth = 3, linestyle=:dashdotdot)
				#plot!(mean(norm_energy_d[9],dims=2), label= L"\kappa = 2 h^{-1}", linewidth = 3, linestyle=:dot)
				title!("Error norm for scenario IV")
				savefig("$dir/plots/$(date)_kappa2_Y6_homo_N_$(N)_IV_sign.png")
end



	# # never save the solutions INSIDE the git repo, they are too large, please make a folder solutions at the same level as the git repo and save them there
	# jldopen("../../solutions/sol_def_N4.jld2", true, true, true, IOStream) do file
	# 	file["sol1"] = sol1
	# end
	#
	# @save "../../solutions/sol_kp525_ki0005_N4_pn_de-in_Q.jld2" sol1
