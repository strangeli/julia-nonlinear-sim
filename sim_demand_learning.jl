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
	num_days =  10
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

@everywhere begin
	l_day = 3600*24 # DemCurve.l_day
	l_hour = 3600 # DemCurve.l_hour
	l_minute = 60 # DemCurve.l_minute
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=0.2,kP=52,T_inv=1/0.05,kI=10)
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=0.2,kP=525,T_inv=1/0.05,kI=0.005)
	# low_layer_control = system_structs.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner=N),kI=repeat([0.005], inner=N)) # different for each node, change array
	low_layer_control = system_structs.LeakyIntegratorPars(M_inv=[1/5.; 1/4.8; 1/4.1; 1/4.8],kP= [400.; 110.; 100.; 200.],T_inv=[1/0.04; 1/0.045; 1/0.047; 1/0.043],kI=[0.05; 0.004; 0.05; 0.001]) # different for each node, change array
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=[0.1; 10; 100; 1000],T_inv=repeat([1/0.05], inner=N),kI=repeat([0.005], inner=N)) # different for each node, change array
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=[1/0.05; 1/0.5; 1/5; 1/50],kI=repeat([0.005], inner=N)) # different for each node, change array
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner = N),kI=[0.005; 0.5; 5; 500]) # different for each node, change array
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=[0.002; 0.2; 2; 20],kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner = N),kI=repeat([0.005], inner=N)) # different for each node, change array
	kappa = 1.0 / l_hour
end

############################################
# NETWORK - this should only run on one process
############################################

# # Full graph for N=4 and degree 3 graph otherwise, change last 3 to 1 for N=2
begin
	graph = random_regular_graph(iseven(3N) ? N : (N-1), 3)
end

# change last "3" to 1 for N=2

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

# fixed amp over the days
# demand_amp = rand(N) .* 250.

# # slowly increasing amplitude - only working for 10 days now
# demand_ampp = demand_amp_var(repeat([10 20 30 40 50 60 70 80 90 100 110], outer=Int(N/2))') # random positive amp over days by 10%
# demand_ampn = demand_amp_var(repeat([-10 -20 -30 -40 -50 -60 -70 -80 -90 -100 -110], outer=Int(N/2))') # random positive amp over days by 10%
# demand_amp = t->vcat(demand_ampp(t), demand_ampn(t))

# # slowly decreasing amplitude - only working for 10 days now
# demand_ampp = demand_amp_var(repeat([110 100 90 80 70 60 50 40 30 20 10], outer=Int(N/2))') # random positive amp over days by 10%
# demand_ampn = demand_amp_var(repeat([-110 -100 -90 -80 -70 -60 -50 -40 -30 -20 -10], outer=Int(N/2))') # random positive amp over days by 10%
# demand_amp = t->vcat(demand_ampp(t), demand_ampn(t))

# # slowly decreasing and increasing amplitude - only working for 10 days now
# demand_ampp = demand_amp_var(repeat([120 120 120 120 120 170 200 120 120 120 120 120 170 200 120], outer=Int(N/2))') # random positive amp over days by 10%
# demand_ampn = demand_amp_var(repeat([120 120 120 120 120 170 200 120 120 120 120 120 170 200 120], outer=Int(N/2))') # random positive amp over days by 10%
# demand_amp = t->vcat(demand_ampp(t), demand_ampn(t))

# slowly increasing and decreasing amplitude - only working for <= 10 days and N = 4 now
demand_amp1 = demand_amp_var(repeat([80 80 80 10 10 10 40 40 40 40 40], outer=Int(N/4))') # random positive amp over days by 10%
demand_amp2 = demand_amp_var(repeat([10 10 10 80 80 80 40 40 40 40 40], outer=Int(N/4))') # random positive amp over days by 10%
demand_amp3 = demand_amp_var(repeat([60 60 60 60 10 10 10 40 40 40 40], outer=Int(N/4))') # random positive amp over days by 10%
demand_amp4 = demand_amp_var(repeat([30 30 30 30 10 10 10 80 80 80 80], outer=Int(N/4))') # random positive amp over days by 10%
demand_amp = t->vcat(demand_amp1(t), demand_amp2(t), demand_amp3(t), demand_amp4(t))


# # random positive amp over days by 30%
# demand_ampp = demand_amp_var(70 .+ rand(num_days+1,Int(N/2)).* 30.)
# demand_ampn = demand_amp_var(70 .+ rand(num_days+1,Int(N/2)).* 30.)  # random negative amp over days by 10%
# demand_amp = t->vcat(demand_ampp(t), demand_ampn(t))

periodic_demand =  t-> demand_amp(t)./100 .* sin(t*pi/(24*3600))^2
samples = 24*4

inter = interpolate([.2 * randn(N) for i in 1:(num_days * samples + 1)], BSpline(Linear()))
residual_demand = t -> inter(1. + t / (24*3600) * samples) # 1. + is needed to avoid trying to access out of range

#########################################
#            SIM                      #
#########################################


vc1 = 1:N # ilc_nodes (here: without communication)
cover1 = Dict([v => [] for v in vc1])# ilc_cover
u = [zeros(1000,1);1;zeros(1000,1)];
fc = 1/6;
a = digitalfilter(Lowpass(fc),Butterworth(2));
Q1 = filtfilt(a,u);# Markov Parameter
Q = Toeplitz(Q1[1001:1001+24-1],Q1[1001:1001+24-1]);


compound_pars = system_structs.compound_pars(N, low_layer_control, kappa, vc1, cover1, Q)
compound_pars.hl.daily_background_power .= 0
compound_pars.hl.current_background_power .= 0
compound_pars.hl.mismatch_yesterday .= 0.
compound_pars.periodic_demand  = periodic_demand # t -> zeros(N) #periodic_demand
compound_pars.residual_demand = residual_demand #t -> zeros(N) #residual_demand
compound_pars.graph = graph
coupfact= 6.
compound_pars.coupling = coupfact .* diagm(0=>ones(ne(graph)))


begin
	factor = 0#0.01*rand(compound_pars.D * compound_pars.N)#0.001#0.00001
	ic = factor .* ones(compound_pars.D * compound_pars.N)
	tspan = (0., num_days * l_day)
	ode_tl1 = ODEProblem(network_dynamics.ACtoymodel!, ic, tspan, compound_pars,
	callback=CallbackSet(PeriodicCallback(network_dynamics.HourlyUpdate(), l_hour),
						 PeriodicCallback(network_dynamics.DailyUpdate_X, l_day)))
end

sol1 = solve(ode_tl1, Rodas4())


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
for j = 1:N
	ILC_power[2,:,j] = Q*(zeros(24,1) +  kappa*hourly_energy[1:24,j])
end
norm_energy_d = zeros(num_days,N)
for j = 1:N
	norm_energy_d[1,j] = norm(hourly_energy[1:24,j])
end


for i=2:num_days
	for j = 1:N
		ILC_power[i+1,:,j] = Q*(ILC_power[i,:,j] +  kappa*hourly_energy[(i-1)*24+1:i*24,j])
		norm_energy_d[i,j] = norm(hourly_energy[(i-1)*24+1:i*24,j])
	end
end

#ILC_power_agg = maximum(mean(ILC_power.^2,dims=3),dims=2)
ILC_power_agg = [norm(mean(ILC_power,dims=3)[d,:]) for d in 1:num_days+2]
ILC_power_hourly_mean = vcat(mean(ILC_power,dims=3)[:,:,1]'...)
ILC_power_hourly_mean_node1 = vcat(ILC_power[:,:,1]'...)
ILC_power_hourly = [norm(reshape(ILC_power,(num_days+2)*24,N)[h,:]) for h in 1:24*(num_days+2)]
ILC_power_hourly_node1 = [norm(reshape(ILC_power,(num_days+2)*24,N)[h,1]) for h in 1:24*(num_days+2)]
dd = t->((periodic_demand(t) .+ residual_demand(t))./100)
load_amp = [first(maximum(dd(t))) for t in 1:3600*24:3600*24*num_days]

norm_hourly_energy = [norm(hourly_energy[h,:]) for h in 1:24*num_days]

using LaTeXStrings

# NODE WISE second-wisenode = 1
node = 1
p1 = plot()
ILC_power_hourly_mean_node = vcat(ILC_power[:,:,node]'...)
dd = t->((periodic_demand(t) .+ residual_demand(t)))
plot!(0:num_days*l_day, t -> dd(t)[node], alpha=0.2, label = latexstring("P^d_$node"),linewidth=3, linestyle=:dot)
plot!(1:3600:24*num_days*3600,hourly_energy[1:num_days*24,node]./3600, label=latexstring("y_$node^{c,h}"),linewidth=3) #, linestyle=:dash)
plot!(1:3600:num_days*24*3600,  ILC_power_hourly_mean_node[1:num_days*24], label=latexstring("\$u_$node^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
               xtickfontsize=14,
    		   legendfontsize=10, linewidth=3, yaxis=("normed power",font(14)),legend=false, lc =:black, margin=5Plots.mm)
ylims!(-0.7,1.5)
title!(L"j = 1")
savefig("$dir/plots/demand_seconds_Y$(coupfact)_node_$(node)_hetero.png")

node = 2
p2 = plot()
ILC_power_hourly_mean_node = vcat(ILC_power[:,:,node]'...)
dd = t->((periodic_demand(t) .+ residual_demand(t)))
plot!(0:num_days*l_day, t -> dd(t)[node], alpha=0.2, label = latexstring("P^d_$node"),linewidth=3, linestyle=:dot)
plot!(1:3600:24*num_days*3600,hourly_energy[1:num_days*24,node]./3600, label=latexstring("y_$node^{c,h}"),linewidth=3)#, linestyle=:dash)
plot!(1:3600:num_days*24*3600,  ILC_power_hourly_mean_node[1:num_days*24], label=latexstring("\$u_$node^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
               xtickfontsize=14, yticks=false, #xaxis=("days [c]",font(14)), yaxis=("normed power",font(14))
    		   legendfontsize=10, linewidth=3,legend=false, lc =:black, margin=5Plots.mm)
ylims!(-0.7,1.5)
title!(L"j = 2")
savefig("$dir/plots/demand_seconds_Y$(coupfact)_node_$(node)_hetero.png")

node = 3
p3 = plot()
ILC_power_hourly_mean_node = vcat(ILC_power[:,:,node]'...)
dd = t->((periodic_demand(t) .+ residual_demand(t)))
plot!(0:num_days*l_day, t -> dd(t)[node], alpha=0.2, label = latexstring("P^d_$node"),linewidth=3, linestyle=:dot)
plot!(1:3600:24*num_days*3600,hourly_energy[1:num_days*24,node]./3600, label=latexstring("y_$node^{c,h}"),linewidth=3)#, linestyle=:dash)
plot!(1:3600:num_days*24*3600,  ILC_power_hourly_mean_node[1:num_days*24], label=latexstring("\$u_$node^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
               xtickfontsize=18,
    		   legendfontsize=10, linewidth=3,xaxis=("days [c]",font(14)),yaxis=("normed power",font(14)),legend=false, lc =:black, margin=5Plots.mm)
ylims!(-0.7,1.5)
title!(L"j = 3")
savefig("$dir/plots/demand_seconds_Y$(coupfact)_node_$(node)_hetero.png")

node = 4
p4 = plot()
ILC_power_hourly_mean_node = vcat(ILC_power[:,:,node]'...)
dd = t->((periodic_demand(t) .+ residual_demand(t)))
plot!(0:num_days*l_day, t -> dd(t)[node], alpha=0.2, label = latexstring("P^d_$node"),linewidth=3, linestyle=:dot)
plot!(1:3600:24*num_days*3600,hourly_energy[1:num_days*24,node]./3600, label=latexstring("y_$node^{c,h}"),linewidth=3)#, linestyle=:dash)
plot!(1:3600:num_days*24*3600,  ILC_power_hourly_mean_node[1:num_days*24], label=latexstring("\$u_$node^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
               xtickfontsize=18,
    		   legendfontsize=10, yticks=false, linewidth=3,xaxis=("days [c]",font(14)),legend=false, lc =:black, margin=5Plots.mm)
ylims!(-0.7,1.5)
title!(L"j = 4")
savefig("$dir/plots/demand_seconds_Y$(coupfact)_node_$(node)_hetero.png")

l = @layout [a b; c d]
plot_demand = plot(p1,p2,p3,p4,layout = l)
savefig(plot_demand, "$dir/plots/demand_seconds_Y$(coupfact)_all_nodes_hetero.png")

l2 = @layout [a b]
plot_demand2 = plot(p2,p4,layout = l2)
savefig(plot_demand2, "$dir/plots/demand_seconds_Y$(coupfact)_nodes2+4_hetero.png")

psum = plot()
ILC_power_hourly_mean_sum = vcat(ILC_power[:,:,1]'...) .+ vcat(ILC_power[:,:,2]'...) .+ vcat(ILC_power[:,:,3]'...) .+ vcat(ILC_power[:,:,4]'...)
dd = t->((periodic_demand(t) .+ residual_demand(t)))
plot!(0:num_days*l_day, t -> (dd(t)[1] .+ dd(t)[2] .+ dd(t)[3] .+ dd(t)[4]), alpha=0.2, label = latexstring("\$P^d_j\$"),linewidth=3, linestyle=:dot)
plot!(1:3600:24*num_days*3600,(hourly_energy[1:num_days*24,1] + hourly_energy[1:num_days*24,2] + hourly_energy[1:num_days*24,3] + hourly_energy[1:num_days*24,4])./3600, label=latexstring("y_j^{c,h}"),linewidth=3, linestyle=:dash)
plot!(1:3600:num_days*24*3600,  ILC_power_hourly_mean_sum[1:num_days*24], label=latexstring("\$u_j^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
               xtickfontsize=18,legend=false,
    		   legendfontsize=10, linewidth=3,xaxis=("days [c]",font(14)),  yaxis=("normed power",font(14)),lc =:black, margin=5Plots.mm)
#ylims!(-0.7,1.5)
#title!("Initial convergence")
savefig(psum,"$dir/plots/demand_seconds_Y$(coupfact)_sum_hetero.png")




# hourly plotting
using LaTeXStrings
plot(0:(num_days)*24-1, ILC_power_hourly[1:num_days*24], label=L"$\max_h \Vert P_{ILC, k}\Vert$", xticks = (1:24:24*num_days, string.(1:num_days)))
plot!(0:24*num_days-1,norm_hourly_energy./3600, label=L"y_h")
plot!(0:24:24*num_days-1, load_amp, label = "demand amplitude")
xlabel!("hour h [h]")
ylabel!("normed quantities [a.u.]")
savefig("$dir/plots/yh_demand_ILC_new_hourly_hetero.png")


# daily plotting
plot(1:num_days, ILC_power_agg[1:num_days,1,1] ./ maximum(ILC_power_agg), label=L"$\max_h \Vert P_{ILC, k}\Vert$")
plot!(1:num_days, mean(norm_energy_d,dims=2) ./ maximum(norm_energy_d), label=L"norm(y_h)")
plot!(1:num_days, load_amp  ./ maximum(load_amp), label = "demand amplitude")
xlabel!("day d [d]")
ylabel!("normed quantities [a.u.]")
savefig("$dir/plots/demand_daily_hetero.png")
