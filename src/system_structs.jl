@doc """
This is a module that contains functions with specified setups for numerical
experiments.
It depends on the modules `network_dynamics.jl` and `observables.jl`

# Examples
```julia-repl
julia> include("src/system_structs.jl")
Main.system_structs
```
"""
module system_structs

	using Random # fix random seed
	using DifferentialEquations # problems types
	using Plots # custom plot functions
	using LightGraphs # create incidence matrices
	using Parameters
	using LinearAlgebra
	using Interpolations
	using StatsBase
	using Statistics
	using Distributions
	using DSP
	using ToeplitzMatrices


	begin
		# import dynamics and observables
		dir = @__DIR__
		include("$dir/network_dynamics.jl")
		include("$dir/observables.jl")
	end

	l_hour = 60 * 60 # in s
	l_day = l_hour * 24 # in s

	@doc """
	    LeakyIntegratorPars(M_inv, kP, T_inv, kI)
	Define a parameter struct.
	"""
	@with_kw mutable struct LeakyIntegratorPars
	    M_inv
	    kP
	    T_inv
		kI
	end

	@with_kw mutable struct ILCPars
		kappa
		mismatch_yesterday#::Array{Float64, 2}
		daily_background_power#::Array{Float64, 2} # 24xN vector with the background power for each hour.
		current_background_power
		ilc_nodes
		ilc_covers
		Q
		lambda
		sign_save
	end



	@doc """
	    UlmoPars(N, D, ll. hl, periodic_infeed, periodic_demand, fluctuating_infeed, residual_demand, incidence, coupling, graph)
	Define a parameter struct.
	"""
	@with_kw mutable struct UlMoPars
		N::Int # number of nodes
		D::Int # degrees of freedom of each node
	    ll::LeakyIntegratorPars # low level control parameters
		hl:: ILCPars			# high level control parameters
		periodic_infeed
		periodic_demand
		fluctuating_infeed
		residual_demand
		incidence
		coupling
		graph
		"""
		Constructor
		"""
		function UlMoPars(N::Int, #D::Int, # degrees of freedom of each node
	    					ll::LeakyIntegratorPars,
							hl:: ILCPars,
							periodic_infeed,
							periodic_demand,
							fluctuating_infeed,
							residual_demand,
							#incidence,
							coupling,
							graph)
				new(N, 5,
				ll,
				hl,
				periodic_infeed,
				periodic_demand,
				fluctuating_infeed,
				residual_demand,
				incidence_matrix(graph,oriented=true),
				coupling,
				graph)
		end
	end

	modx(x, y) = (mod2pi.(x .+ π/2) .- π/2, y)
	mody(x, y) = (x, mod2pi.(y .+ π/2) .- π/2)


	@doc """
	    default_pars(N)
	Setup the system with default parameters.
	"""
	function default_pars(N)
		low_layer_control = LeakyIntegratorPars(M_inv=60.,kP=1.3,T_inv=1.,kI=0.9)
		#g = SimpleGraph(1)
		g = random_regular_graph(iseven(3N) ? N : (N-1), 3)
		#incidence = incidence_matrix(g, oriented=true)
		coupling = 6. .* diagm(0=>ones(size(incidence_matrix(g, oriented=true),2)))
		vc = 1:N # independent_set(g, DegreeIndependentSet()) # ilc_nodes
		cover = [] # Dict([v => neighbors(g, v) for v in vc]) # ilc_cover
		u = [zeros(1000,1);1;zeros(1000,1)];
		fc = 1/6;
		a = digitalfilter(Lowpass(fc),Butterworth(2));
		Q1 = filtfilt(a,u);#Markov Parameter
		Q = Toeplitz(Q1[1001:1001+24-1],Q1[1001:1001+24-1]);
		lambda = 1.
		sign_save = zeros(1,N+1)
		higher_layer_control = ILCPars(kappa=0.35, mismatch_yesterday=zeros(24, N), daily_background_power=zeros(24, N), current_background_power=zeros(N), ilc_nodes=vc, ilc_covers=cover, Q=Q, lambda = lambda, sign_save = sign_save)
		periodic_infeed = t -> zeros(N)
		peak_demand = rand(N)
		periodic_demand= t -> zeros(N)#peak_demand .* abs(sin(pi * t/24.))
		fluctuating_infeed = t -> zeros(N)
		residual_demand= t -> zeros(N)



		return UlMoPars(N, low_layer_control,
								higher_layer_control,
								periodic_infeed,
								periodic_demand,
								fluctuating_infeed,
								residual_demand,
								#incidence,
								coupling,
								g)
	end

	function compound_pars(N, low_layer_control, kappa, ilc_nodes, ilc_covers, Q, lambda)
		sign_save = zeros(1,N+1)

		higher_layer_control = ILCPars(kappa=kappa, mismatch_yesterday=zeros(24, N), daily_background_power=zeros(24, N), current_background_power=zeros(N),ilc_nodes=ilc_nodes, ilc_covers=ilc_covers, Q=Q, lambda = lambda, sign_save = sign_save)

		periodic_infeed = t -> zeros(N)
		periodic_demand= t -> zeros(N)
		fluctuating_infeed = t -> zeros(N)
		residual_demand = t -> zeros(N)

		# make sure N*k is even, otherwise the graph constructor fails
		#g = SimpleGraph(1)
		g = random_regular_graph(iseven(3N) ? N : (N-1), 3)
		#incidence = incidence_matrix(g, oriented=true)
		coupling= 6. .* diagm(0=>ones(ne(g)))

		return UlMoPars(N, low_layer_control,
								higher_layer_control,
								periodic_infeed,
								periodic_demand,
								fluctuating_infeed,
								residual_demand,
								#incidence,
								coupling,
								g)
	end

	##############################################################################
	##############################################################################
	# Monte Carlo functions

	get_run(i, batch_size) = mod(i, batch_size)==0 ? batch_size : mod(i, batch_size)
	get_batch(i, batch_size) = 1 + (i - 1) ÷ batch_size


	function prob_func_ic(prob, i, repeat, batch_size, kappa_lst, num_days)
		println("sim ", i)
		run = get_run(i, batch_size)
	    batch = get_batch(i, batch_size)

		prob.p.hl.daily_background_power .= 0.
		prob.p.hl.current_background_power .= 0.
		prob.p.hl.mismatch_yesterday .= 0.

		prob.p.hl.kappa = kappa_lst[batch]

		#prob.p.coupling = 800. .* diagm(0=>ones(ne(prob.p.graph)))

		hourly_update = network_dynamics.HourlyUpdate()

		ODEProblem(network_dynamics.ACtoymodel!, prob.u0, prob.tspan, prob.p,
			callback=CallbackSet(PeriodicCallback(hourly_update, 3600),
								 PeriodicCallback(network_dynamics.DailyUpdate_X, 3600*24)))
	end



	function observer_ic(sol, i, freq_filter, energy_filter, freq_threshold, num_days,N) # what should be extracted from one run
		# sol.prob.callback.discrete_callbacks[1].affect!.f.integrated_control_power_history
		omega_max = maximum(abs.(sol[freq_filter,:]))
		ex = observables.frequency_exceedance(sol, freq_filter, freq_threshold)
		#control_energy = observables.sum_abs_energy_last_days(sol, energy_filter, sol.prob.tspan[2]/(24*3600))
		control_energy = observables.sum_abs_energy_last_days(sol, energy_filter, sol.prob.tspan[2]/(24*3600))
		var_omega = var(sol,dims=2)[freq_filter]
		#var_ld = observables.var_last_days(sol, freq_filter, sol.prob.tspan[2]/(24*3600))
		var_ld = observables.var_last_days(sol, freq_filter, sol.prob.tspan[2]/(24*3600))
	#	control_energy_abs = sol[energy_abs_filter,end]

		hourly_energy = zeros(24*num_days,N)
		for i=1:24*num_days
			for j = 1:N
				hourly_energy[i,j] = sol(i*3600)[energy_filter[j]]
			end
		end

		ILC_power = zeros(num_days,24,N)
		norm_energy_d = zeros(num_days,N)
		norm_energy_dn = zeros(num_days,1)
		for j = 1:N
			norm_energy_d[1,j] = norm(hourly_energy[1:24,j])
		end
		norm_energy_dn[1,1] = norm(hourly_energy[1:24,:])

		for i=2:num_days
			for j = 1:N
				ILC_power[i,:,j] = sol.prob.p.hl.Q*(ILC_power[i-1,:,j] +  sol.prob.p.hl.kappa*hourly_energy[(i-1)*24+1:i*24,j])
			end
			for j = 1:N
				norm_energy_d[i,j] = norm(hourly_energy[(i-1)*24+1:i*24,j])
			end
			norm_energy_dn[i,1] = norm(hourly_energy[(i-1)*24+1:i*24,:])
		end

		((omega_max, ex, control_energy, var_omega, Array(adjacency_matrix(sol.prob.p.graph)), sol.prob.p.hl.kappa, sol.prob.p.hl.ilc_nodes, sol.prob.p.hl.ilc_covers, var_ld, hourly_energy, norm_energy_d, norm_energy_dn), false)
	end



	function reduction_ic(u, data, I, batch_size) # what should be extracted from one batch
	    # u is the solution of previous batches
	    # data is the solution of the current batch
	    # we obtain:
		omega_max_abs = [dat[1] for dat in data] # max frequency
	    	ex = [dat[2] for dat in data] # ex
		energy = [dat[3] for dat in data] # working as array???
		energy_abs = [dat[10] for dat in data]
		#var_omega = [dat[4] for dat in data]
		#var_omega_ld = [dat[5] for dat in data]

		omega_max_max = maximum(omega_max_abs)
		ex_mean = sum(ex)/batch_size
		energy_mean = sum(energy)/length(energy) # summing over all nodes and runs in one batches now
		ex_std = std(ex)
		energy_std = std(energy)
		energy_abs_mean = sum(energy_abs)/length(energy_abs)

		new_output = [omega_max_max, ex_mean, energy_mean, ex_std, energy_std, energy_abs_mean] #var_omega_max var_omega_ld_mean]
		append!(u, [new_output, ]), false # This way of append! ensures that we get an Array of Arrays
	end


	##############################################################################
	# communication scenarios



	function prob_func_I(prob, i, repeat, batch_size, num_days, kappa_lst)
		println("sim ", i)
		run = get_run(i, batch_size)
	    batch = get_batch(i, batch_size)

		prob.p.hl.daily_background_power .= 0.
		prob.p.hl.current_background_power .= 0.
		prob.p.hl.mismatch_yesterday .= 0.
		prob.p.hl.kappa = kappa_lst[batch]


		hourly_update = network_dynamics.HourlyUpdateEcon()
		#hourly_update = network_dynamics.HourlyUpdateEcon()

		ODEProblem(network_dynamics.ACtoymodel!, prob.u0, prob.tspan, prob.p,
			callback=CallbackSet(PeriodicCallback(hourly_update, 3600, initial_affect= false),
								 PeriodicCallback(network_dynamics.DailyUpdate, 3600*24, initial_affect= false)))
	end

	function prob_func_II(prob, i, repeat, batch_size, num_days, kappa_lst)
		println("sim ", i)
		run = get_run(i, batch_size)
	    batch = get_batch(i, batch_size)

		prob.p.hl.daily_background_power .= 0.
		prob.p.hl.current_background_power .= 0.
		prob.p.hl.mismatch_yesterday .= 0.
		prob.p.hl.kappa = kappa_lst[batch]

		hourly_update = network_dynamics.HourlyUpdateEcon()

		ODEProblem(network_dynamics.ACtoymodel!, prob.u0, prob.tspan, prob.p,
			callback=CallbackSet(PeriodicCallback(hourly_update, 3600, initial_affect= false),
								 PeriodicCallback(network_dynamics.DailyUpdate_II, 3600*24, initial_affect= false)))
	end

	function prob_func_III(prob, i, repeat, batch_size, num_days, kappa_lst)
		println("sim ", i)
		run = get_run(i, batch_size)
	    batch = get_batch(i, batch_size)

		prob.p.hl.daily_background_power .= 0.
		prob.p.hl.current_background_power .= 0.
		prob.p.hl.mismatch_yesterday .= 0.
		prob.p.hl.kappa = kappa_lst[batch]

		hourly_update = network_dynamics.HourlyUpdateEcon()

		ODEProblem(network_dynamics.ACtoymodel!, prob.u0, prob.tspan, prob.p,
			callback=CallbackSet(PeriodicCallback(hourly_update, 3600, initial_affect= false),
								 PeriodicCallback(network_dynamics.DailyUpdate_III, 3600*24, initial_affect= false)))
	end


	function prob_func_IV(prob, i, repeat, batch_size, num_days, kappa_lst)
		println("sim ", i)
		run = get_run(i, batch_size)
	    batch = get_batch(i, batch_size)


		prob.p.hl.daily_background_power .= 0.
		prob.p.hl.current_background_power .= 0.
		prob.p.hl.mismatch_yesterday .= 0.
		prob.p.hl.kappa = kappa_lst[batch]


		hourly_update = network_dynamics.HourlyUpdateEcon()

		ODEProblem(network_dynamics.ACtoymodel!, prob.u0, prob.tspan, prob.p,
			callback=CallbackSet(PeriodicCallback(hourly_update, 3600, initial_affect= false),
								 PeriodicCallback(network_dynamics.DailyUpdate_IV, 3600*24, initial_affect= false)))
	end

	function observer_basic_types(sol, i, energy_filter) # what should be extracted from one run
		N = sol.prob.p.N
		num_days = Int(sol.prob.tspan[2]/(24*3600))

		hourly_energy = zeros(24*num_days,N)

		for i=1:24*num_days
			for j = 1:N
				hourly_energy[i,j] = sol(i*3600)[energy_filter[j]]
			end
		end

		norm_energy_d = zeros(num_days,N)
		norm_energy_dn = zeros(num_days,1)

		for i=1:num_days
			norm_energy_dn[i,1] = norm(hourly_energy[(i-1)*24+1:i*24,:])
			for j = 1:N
				norm_energy_d[i,j] = norm(hourly_energy[(i-1)*24+1:i*24,j])
			end
		end

		((sol.prob.p.hl.kappa, hourly_energy, norm_energy_d, norm_energy_dn), false)
	end





	function observer_basic_types_I(sol, i, freq_filter, energy_filter, energy_abs_filter, freq_threshold, obs_days) # what should be extracted from one run
		N = sol.prob.p.N
		num_days = Int(sol.prob.tspan[2]/(24*3600))
		omega_max = maximum(abs.(sol(sol.prob.tspan[2]-obs_days*3600*24:sol.prob.tspan[2])[freq_filter,:]))
		ex = observables.frequency_exceedance(sol, freq_filter, freq_threshold)
		control_energy = observables.sum_abs_energy_last_days(sol, energy_filter, obs_days)
		var_omega = var(sol,dims=2)[freq_filter]
		var_ld = observables.var_last_days(sol, freq_filter, obs_days)
		hourly_energy = zeros(24*num_days,N)
		for i=1:24*num_days
			for j = 1:N
				hourly_energy[i,j] = sol(i*3600)[energy_filter[j]]
			end
		end

		ILC_power = zeros(num_days,24,N)
		norm_energy_d = zeros(num_days,N)
		for j = 1:N
			norm_energy_d[1,j] = norm(hourly_energy[1:24,j])
		end

		for i=2:num_days
			for j = 1:N
				ILC_power[i,:,j] = (ILC_power[i-1,:,j] +  sol.prob.p.hl.kappa*hourly_energy[(i-1)*24+1:i*24,j])
			end
			for j = 1:N
				norm_energy_d[i,j] = norm(hourly_energy[(i-1)*24+1:i*24,j])
			end
		end

		((omega_max, ex, control_energy, var_omega, Array(adjacency_matrix(sol.prob.p.graph)), sol.prob.p.hl.kappa, sol.prob.p.hl.ilc_nodes, sol.prob.p.hl.ilc_covers, var_ld, hourly_energy, norm_energy_d), false)
	end


	function observer_basic_types_II(sol, i, freq_filter, energy_filter, energy_abs_filter, freq_threshold, obs_days) # what should be extracted from one run
		N = sol.prob.p.N
		num_days = Int(sol.prob.tspan[2]/(24*3600))
		omega_max = maximum(abs.(sol(sol.prob.tspan[2]-obs_days*3600*24:sol.prob.tspan[2])[freq_filter,:]))
		ex = observables.frequency_exceedance(sol, freq_filter, freq_threshold)
		control_energy = observables.sum_abs_energy_last_days(sol, energy_filter, obs_days)
		var_omega = var(sol,dims=2)[freq_filter]
		var_ld = observables.var_last_days(sol, freq_filter, obs_days)

				hourly_energy = zeros(24*num_days,N)
				for i=1:24*num_days
					for j = 1:N
						hourly_energy[i,j] = sol(i*3600)[energy_filter[j]]
					end
				end

				ILC_power = zeros(num_days,24,N)
				norm_energy_d = zeros(num_days,N)
				for j = 1:N
					norm_energy_d[1,j] = norm(hourly_energy[1:24,j])
				end

				for i=2:num_days
					for j = 1:N
						norm_energy_d[i,j] = norm(hourly_energy[(i-1)*24+1:i*24,j])
					end
				end


		((omega_max, ex, control_energy, var_omega, Array(adjacency_matrix(sol.prob.p.graph)), sol.prob.p.hl.kappa, sol.prob.p.hl.ilc_nodes, sol.prob.p.hl.ilc_covers, var_ld,  hourly_energy, norm_energy_d), false)
	end

	function observer_basic_types_III(sol, i, freq_filter, energy_filter, energy_abs_filter, freq_threshold, obs_days) # what should be extracted from one run
		N = sol.prob.p.N
		num_days = Int(sol.prob.tspan[2]/(24*3600))
		omega_max = maximum(abs.(sol(sol.prob.tspan[2]-obs_days*3600*24:sol.prob.tspan[2])[freq_filter,:]))
		ex = observables.frequency_exceedance(sol, freq_filter, freq_threshold)
		control_energy = observables.sum_abs_energy_last_days(sol, energy_filter, obs_days)
		var_omega = var(sol,dims=2)[freq_filter]
		var_ld = observables.var_last_days(sol, freq_filter, obs_days)

				hourly_energy = zeros(24*num_days,N)
				for i=1:24*num_days
					for j = 1:N
						hourly_energy[i,j] = sol(i*3600)[energy_filter[j]]
					end
				end

				norm_energy_d = zeros(num_days,N)
				for j = 1:N
					norm_energy_d[1,j] = norm(hourly_energy[1:24,j])
				end

				for i=2:num_days
					for j = 1:N
						norm_energy_d[i,j] = norm(hourly_energy[(i-1)*24+1:i*24,j])
					end
				end

		((omega_max, ex, control_energy, var_omega, Array(adjacency_matrix(sol.prob.p.graph)), sol.prob.p.hl.kappa, sol.prob.p.hl.ilc_nodes, sol.prob.p.hl.ilc_covers, var_ld, hourly_energy, norm_energy_d), false)
	end

	function observer_basic_types_IV(sol, i, freq_filter, energy_filter, energy_abs_filter, freq_threshold, obs_days) # what should be extracted from one run
		N = sol.prob.p.N
		num_days = Int(sol.prob.tspan[2]/(24*3600))
		omega_max = maximum(abs.(sol(sol.prob.tspan[2]-obs_days*3600*24:sol.prob.tspan[2])[freq_filter,:]))
		ex = observables.frequency_exceedance(sol, freq_filter, freq_threshold)
		control_energy = observables.sum_abs_energy_last_days(sol, energy_filter, obs_days)
		var_omega = var(sol,dims=2)[freq_filter]
		var_ld = observables.var_last_days(sol, freq_filter, obs_days)
						hourly_energy = zeros(24*num_days,N)
				for i=1:24*num_days
					for j = 1:N
						hourly_energy[i,j] = sol(i*3600)[energy_filter[j]]
					end
				end

				norm_energy_d = zeros(num_days,N)
				for j = 1:N
					norm_energy_d[1,j] = norm(hourly_energy[1:24,j])
				end

				for i=2:num_days
					for j = 1:N
						norm_energy_d[i,j] = norm(hourly_energy[(i-1)*24+1:i*24,j])
					end
				end

		((omega_max, ex, control_energy, var_omega, Array(adjacency_matrix(sol.prob.p.graph)), sol.prob.p.hl.kappa, sol.prob.p.hl.ilc_nodes, sol.prob.p.hl.ilc_covers, var_ld, hourly_energy, norm_energy_d), false)
	end



	################################################
	# plotting help

	function movingmean(t,int,a,b)
	    idx = findall(x -> t + int > x > t - int, a)
	    mean(b[idx])
	end

end
