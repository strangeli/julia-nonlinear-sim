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

begin
	l_day = 3600*24 # DemCurve.l_day
	l_hour = 3600 # DemCurve.update
	update = l_hour/4 #/2 for half # DemCurve.update
	n_updates_per_day=96 # 24 l_day / update
	l_minute = 60 # DemCurve.l_minute
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=0.2,kP=52,T_inv=1/0.05,kI=10)
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=0.2,kP=525,T_inv=1/0.05,kI=0.005)
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner=N),kI=repeat([0.005], inner=N)) # different for each node, change array
	low_layer_control = system_structs.LeakyIntegratorPars(M_inv=[1/5.; .8; .1; .8],kP= [400.; 110.; 100.; 200.],T_inv=[1/0.04; 1/0.045; 1/0.047; 1/0.043],kI=[0.05; 0.004; 0.05; 0.001]) # different for each node, change array
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=[0.1; 10; 100; 1000],T_inv=repeat([1/0.05], inner=N),kI=repeat([0.005], inner=N)) # different for each node, change array
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=[1/0.05; 1/0.5; 1/5; 1/50],kI=repeat([0.005], inner=N)) # different for each node, change array
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner = N),kI=[0.005; 0.5; 5; 500]) # different for each node, change array
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=[0.002; 0.2; 2; 20],kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner = N),kI=repeat([0.005], inner=N)) # different for each node, change array
	kappa =(1.0 / update) #*2 for update half
end

############################################
# NETWORK - this should only run on one process
############################################

# # Full graph for N=4 and degree 3 graph otherwise, change last 3 to 1 for N=2
begin
	graph = random_regular_graph(iseven(3N) ? N : (N-1), 3)
end



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
Q = Toeplitz(Q1[1001:1001+n_updates_per_day-1],Q1[1001:1001+n_updates_per_day-1]);


compound_pars = system_structs.compound_pars(N, low_layer_control, kappa, vc1, cover1, Q, update)
compound_pars.hl.daily_background_power .= 0
compound_pars.hl.current_background_power .= 0
compound_pars.hl.mismatch_yesterday .= 0.
# compound_pars.hl.update = update
compound_pars.periodic_demand  = periodic_demand # t -> zeros(N) #periodic_demand
compound_pars.residual_demand = residual_demand #t -> zeros(N) #residual_demand
compound_pars.graph = graph
coupfact= 6.
compound_pars.coupling = coupfact .* diagm(0=>ones(ne(graph)))

using DataFrames, StatsPlots , StringEncodings , DataFramesMeta , JLD2


begin
	factor = 0#0.01*rand(compound_pars.D * compound_pars.N)#0.001#0.00001
	ic = factor .* ones(compound_pars.D * compound_pars.N)
	tspan = (0., num_days * l_day)
	ode_tl1 = ODEProblem(network_dynamics.ACtoymodel!, ic, tspan, compound_pars,
	callback=CallbackSet(PeriodicCallback(network_dynamics.Updating(),update),
						 PeriodicCallback(network_dynamics.DailyUpdate_PD, l_day)))
end
sol1 = solve(ode_tl1, Rodas4())

using Dates , GraphIO
date = Dates.Date(Dates.now())

if isdir("$dir/solutions/$(date)") == false
	mkdir("$dir/solutions/$(date)")
end

jldopen("$dir/solutions/2020-10-02/sim_demand_learning_pd.jld2", true, true, true, IOStream) do file
	file["t"] = sol1.t
    file["u"] = sol1.u
end

f = jldopen("$dir/solutions/2020-09-26/sim_demand_learning_p.jld2", "r")


using CSV
using JLD2 , Pandas
#JLD2.@save "outputt.jld2" sol1

##################### [3.5841198212278796e-10, 3.078381669806953e-10, -7.730920518542944e-12, 3.5113574636468243e-10, 4.18391402683452e-6, 3.5993826772601687e-6, -9.03894389117365e-8, 4.10344664907011e-6, -8.959660667241752e-9, -6.840813496031148e-9, 1.6447769689790595e-10, -8.16593676270488e-9, -1.4336530395777753e-7, -3.3862588282699924e-8, 7.731014271411215e-10, -7.022761484130914e-8, 1.3708213968165875e-7, 3.2382926029415346e-8, 7.731014271411215e-10, 6.715593233297654e-8]##################################################
#                               PLOTTING                             #
###################''''''''''''''''''''''''###################################################
using Plots
#update_energy= zeros(n_updates_per_day*num_days+1,N)
#for i=1:n_updates_per_day*num_days+1
#	for j = 1:N
#		update_energy[i,j] = sol1((i-1)*update)[energy_filter[j]]
#	end
#end


update_energy_pd_mismatch_yesterday = zeros(n_updates_per_day*num_days+1,N)
for i=1:n_updates_per_day*num_days+1
	for j = 1:N
		update_energy_pd_mismatch_yesterday[i,j] = sol1((i-1)*update)[energy_filter[j]]
	end
end
update_energy_pd_mismatch_d_control = zeros(n_updates_per_day*num_days+1,N)
for i=2:n_updates_per_day*num_days+1
	for j = 1:N
		update_energy_pd_mismatch_d_control[i,j] = sol1((i-1)*update)[energy_filter[j]]
	end
end



update_energy= zeros(n_updates_per_day*num_days+1,N)

for i=1:n_updates_per_day*num_days+1
	for j = 1:N
		for k in 1:length(f["t"])
		  	if (f["t"][k] ==(i-1)*update)
					if (f["u"][k][energy_filter[j]] != 0)
				       update_energy[i,j] = f["u"][k][energy_filter[j]]
				   end
		  	end
		end
	end
 end


# update_energy_pd_mismatch_d_control = zeros(n_updates_per_day*num_days+1,N)
# for i=2:n_updates_per_day*num_days+1
# 	for j = 1:N
# 		for k in 1:length(f["t"])
# 		  	if (f["t"][k] ==(i-1)*update)
# 					if (f["u"][k][energy_filter[j]] != 0)
# 				       update_energy_pd_mismatch_d_control[i,j] = f["u"][k][energy_filter[j]]
#				   end
# 		  	end
# 		end
# 	end
# end


 update_energy_pd =zeros(n_updates_per_day*num_days+1,N)
 for i=1:n_updates_per_day*num_days+1
 	for j = 1:N
 		update_energy_pd[i,j] =update_energy_pd_mismatch_yesterday[i,j]+ (1/1)*(update_energy_pd_mismatch_yesterday[i,j]-update_energy_pd_mismatch_d_control[i,j])
 	end
 end


Plots.plot()
Plots.plot(update_energy)
Plots.plot!(update_energy_pd)
using Images

savefig("$dir/plots/demand_seconds_hetero_update_energy.png")






ILC_power = zeros(num_days+2,n_updates_per_day,N)
for j = 1:N
	ILC_power[2,:,j] = Q*(zeros(n_updates_per_day,1) +  kappa*update_energy[1:n_updates_per_day,j])
end

norm_energy_d = zeros(num_days,N)
for j = 1:N
	norm_energy_d[1,j] = norm(update_energy[1:n_updates_per_day,j])
end
for i=2:num_days
	for j = 1:N
		ILC_power[i+1,:,j] = Q*(ILC_power[i,:,j] +  kappa*update_energy[(i-1)*n_updates_per_day+1:i*n_updates_per_day,j])
		norm_energy_d[i,j] = norm(update_energy[(i-1)*n_updates_per_day+1:i*n_updates_per_day,j])/update
	end
end



#ILC_power = zeros(num_days+2,n_updates_per_day,N)
#for j = 1:N
#	ILC_power[2,:,j] = Q*(zeros(n_updates_per_day,1) +  kappa*update_energy[1:n_updates_per_day,j]
#	+  kappa*(update_energy[1:n_updates_per_day,j]-update_energy_d[1:n_updates_per_day,j]))
#end


#norm_energy_d = zeros(num_days,N)
#for j = 1:N
#	norm_energy_d[1,j] = norm(update_energy[1:n_updates_per_day,j])
#end
#for i=2:num_days
#	for j = 1:N
#		ILC_power[i+1,:,j] = Q*(ILC_power[i,:,j] +  kappa*update_energy[(i-1)*n_updates_per_day+1:i*n_updates_per_day,j]
#		+  kappa*(update_energy[(i-1)*n_updates_per_day+1:i*n_updates_per_day,j]-update_energy_d[(i-1)*n_updates_per_day+1:i*n_updates_per_day,j]))
#		#norm_energy_d[i,j] = norm(update_energy[(i-1)*n_updates_per_day+1:i*n_updates_per_day,j])
#	end
#end



ILC_power_pd = zeros(num_days+2,n_updates_per_day,N)
for j = 1:N
	ILC_power_pd[2,:,j] = Q*(zeros(n_updates_per_day,1) +  kappa*update_energy_pd_mismatch_yesterday[1:n_updates_per_day,j]
	+kappa*(update_energy_pd_mismatch_yesterday[1:n_updates_per_day,j]-update_energy_pd_mismatch_d_control[1:n_updates_per_day,j]))
end




norm_energy_d_pd = zeros(num_days,N)
for j = 1:N
	norm_energy_d_pd[1,j] = norm(update_energy_pd[1:n_updates_per_day,j])
end


for i=2:num_days
	for j = 1:N

		ILC_power_pd[i+1,:,j] = Q*(ILC_power_pd[i,:,j] +  kappa*update_energy_pd_mismatch_yesterday[(i-1)*n_updates_per_day+1:i*n_updates_per_day,j]
		+kappa*(update_energy_pd_mismatch_yesterday[(i-1)*n_updates_per_day+1:i*n_updates_per_day,j]-update_energy_pd_mismatch_d_control[(i-1)*n_updates_per_day+1:i*n_updates_per_day,j]))

		norm_energy_d_pd[i,j] = norm(update_energy_pd[(i-1)*n_updates_per_day+1:i*n_updates_per_day,j])/update
	end
end

Plots.plot()
Plots.plot(ILC_power)
Plots.plot!(ILC_power_pd)


#ILC_power_agg = maximum(mean(ILC_power.^2,dims=3),dims=2)
ILC_power_agg = [norm(mean(ILC_power,dims=3)[d,:]) for d in 1:num_days+2]
ILC_power_update_mean = vcat(mean(ILC_power,dims=3)[:,:,1]'...)
ILC_power_update_mean_node1 = vcat(ILC_power[:,:,1]'...)
ILC_power_update = [norm(reshape(ILC_power,(num_days+2)*n_updates_per_day,N)[h,:]) for h in 1:n_updates_per_day*(num_days+2)]
ILC_power_update_node1 = [norm(reshape(ILC_power,(num_days+2)*n_updates_per_day,N)[h,1]) for h in 1:n_updates_per_day*(num_days+2)]
dd = t->((periodic_demand(t) .+ residual_demand(t))./100)
load_amp = [first(maximum(dd(t))) for t in 1:3600*24:3600*24*num_days]
norm_update_energy = [norm(update_energy[h,:]) for h in 1:n_updates_per_day*num_days]

ILC_power_agg_pd = [norm(mean(ILC_power_pd,dims=3)[d,:]) for d in 1:num_days+2]
ILC_power_update_mean_pd = vcat(mean(ILC_power_pd,dims=3)[:,:,1]'...)
ILC_power_update_mean_node1_pd = vcat(ILC_power_pd[:,:,1]'...)
ILC_power_update_pd = [norm(reshape(ILC_power_pd,(num_days+2)*n_updates_per_day,N)[h,:]) for h in 1:n_updates_per_day*(num_days+2)]
ILC_power_update_node1_pd = [norm(reshape(ILC_power_pd,(num_days+2)*n_updates_per_day,N)[h,1]) for h in 1:n_updates_per_day*(num_days+2)]
dd = t->((periodic_demand(t) .+ residual_demand(t))./100)
load_amp = [first(maximum(dd(t))) for t in 1:3600*24:3600*24*num_days]
norm_update_energy_pd = [norm(update_energy_pd[h,:]) for h in 1:n_updates_per_day*num_days]

using LaTeXStrings
#df = DataFrame(kappa_pd = [p[6] for p in sol1.u] ,
#			   update_energy_pd = [p[10] for p in sol1.u],
#			   norm_energy_d_pd = [p[11] for p in sol1.u],
#			   update_pd = [p[12] for p in sol1.u])

# NODE WISE second-wisenode = 1
node = 1
p1 = Plots.plot()
ILC_power_update_mean_node = vcat(ILC_power[:,:,node]'...)
ILC_power_update_mean_node_pd = vcat(ILC_power_pd[:,:,node]'...)
dd = t->((periodic_demand(t) .+ residual_demand(t)))
Plots.plot(0:num_days*l_day, t -> dd(t)[node], alpha=0.2,legend=false, label = latexstring("P^d"),linewidth=3,legendfontsize=10, linestyle=:dot)


plot!(1:update:24*num_days*3600,(update_energy_pd[1:num_days*n_updates_per_day,node])./update, label=latexstring("y_{pd}"),linewidth=0.3, color ="yellow" ,  w = 2,legend=false,legendfontsize=10,legendfontrotation=10) #, linestyle=:dash)
plot!(1:update:24*num_days*3600,update_energy[1:num_days*n_updates_per_day,node]./update, label=latexstring("y_{p}"),linewidth=0.3,color="red",linestyle=:solid, w = 2)

plot!(1:update:num_days*24*3600,  ILC_power_update_mean_node_pd[1:num_days*n_updates_per_day], label=latexstring("\$u_{pd}^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)),w=2, ytickfontsize=14,
			   	xtickfontsize=18,
			   	legendfontsize=10, linewidth=3, yaxis=("normed power",font(14)),  seriescolor = :orchid,  margin=5Plots.mm)

plot!(1:update:num_days*24*3600,  ILC_power_update_mean_node[1:num_days*n_updates_per_day], label=latexstring("\$u_{p}^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
			   xtickfontsize=18,w=2,
			   legendfontsize=10,linewidth=3, yaxis=("normed power",font(14)), seriescolor =:black, margin=5Plots.mm)


ylims!(-0.7,1.5)
title!(L"j = 1")



#df = DataFrame(update_energy)
#CSV.write("ILC_power_update_mean_node.csv",df)
#ILC_power_update_mean_node_csv = CSV.read("ILC_power_update_mean_node.csv")
#plot(ILC_power_update_mean_node[:,1] , title = "Update energy")
#plot!(1:update:num_days*24*3600,  ILC_power_update_mean_node[1:num_days*n_updates_per_day], label=latexstring("\$u_$node^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
#               xtickfontsize=14,
#    		   legendfontsize=10, linewidth=3, yaxis=("normed power",font(14)),legend=false, lc =:black, margin=5Plots.mm)


savefig("$dir/plots/demand_seconds_Y$(coupfact)_node_$(node)_hetero.png")




node = 2
p2 = Plots.plot()
ILC_power_hourly_mean_node = vcat(ILC_power[:,:,node]'...)
ILC_power_update_mean_node_pd = vcat(ILC_power_pd[:,:,node]'...)

dd = t->((periodic_demand(t) .+ residual_demand(t)))
Plots.plot(0:num_days*l_day, t -> dd(t)[node], alpha=0.2,legend=false, label = latexstring("P^d"),linewidth=3,legendfontsize=10, linestyle=:dot)

plot!(1:update:24*num_days*3600,update_energy_pd[1:num_days*n_updates_per_day,node]./update, label=latexstring("y_{pd}"),linewidth=0.3, color ="yellow",  w = 2,legend=:false,legendfontsize=10,legendfontrotation=10) #, linestyle=:dash)
plot!(1:update:24*num_days*3600,update_energy[1:num_days*n_updates_per_day,node]./update, label=latexstring("y_{p}"),linewidth=0.3,color="red",linestyle=:solid, w = 2)

plot!(1:update:num_days*24*3600,  ILC_power_update_mean_node_pd[1:num_days*n_updates_per_day], label=latexstring("\$u_{pd}^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)),w=2, ytickfontsize=14,
			   	xtickfontsize=18, yticks=false,
			   	legendfontsize=10, linewidth=3,  seriescolor = :orchid,  margin=5Plots.mm)
plot!(1:update:num_days*24*3600,  ILC_power_update_mean_node[1:num_days*n_updates_per_day], label=latexstring("\$u_{p}^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
			   xtickfontsize=18,w=2,yticks=false,
			   legendfontsize=10 ,linewidth=3,  seriescolor =:black, margin=5Plots.mm)


ylims!(-0.7,1.5)
title!(L"j = 2")
savefig("$dir/plots/demand_seconds_Y$(coupfact)_node_$(node)_hetero.png")

node = 3
p3 = Plots.plot()
ILC_power_hourly_mean_node = vcat(ILC_power[:,:,node]'...)
ILC_power_update_mean_node_pd = vcat(ILC_power_pd[:,:,node]'...)

dd = t->((periodic_demand(t) .+ residual_demand(t)))
Plots.plot(0:num_days*l_day, t -> dd(t)[node], alpha=0.2,legend=:false, label = latexstring("P^d"),linewidth=3,legendfontsize=10, linestyle=:dot)

#plot!(1:update:24*num_days*3600,update_energy_pd[1:num_days*n_updates_per_day,node]./update, label=latexstring("y_{pd}"),linewidth=0.3, color ="yellow" ,  w = 2,legend=false,legendfontsize=10,legendfontrotation=10) #, linestyle=:dash)
#plot!(1:update:24*num_days*3600,update_energy[1:num_days*n_updates_per_day,node]./update, label=latexstring("y_{p}"),linewidth=0.3,color="red",linestyle=:solid, w = 2)
plot!(1:update:24*num_days*3600,update_energy_pd[1:num_days*n_updates_per_day,node]./update, label=latexstring("y_$node^{c,h}"),linewidth=3,color ="yellow" )#, linestyle=:dash)
plot!(1:update:24*num_days*3600,update_energy[1:num_days*n_updates_per_day,node]./update, label=latexstring("y_$node^{c,h}"),linewidth=3,color ="red" )#, linestyle=:dash)

#plot!(1:update:num_days*24*3600,  ILC_power_update_mean_node_pd[1:num_days*n_updates_per_day], label=latexstring("\$u_{pd}^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)),w=2, ytickfontsize=14,
#			   	xtickfontsize=18,
#			   	legendfontsize=10, linewidth=3, yaxis=("normed power",font(14)),xaxis=("days [c]",font(14)),  seriescolor = :orchid,  margin=5Plots.mm)


#plot!(1:update:num_days*24*3600,  ILC_power_update_mean_node_pd[1:num_days*n_updates_per_day], label=latexstring("\$u_{pd}^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)),w=2, ytickfontsize=14,
#				xtickfontsize=18, yticks=false,
#				legendfontsize=10, linewidth=3,  seriescolor = :orchid,  margin=5Plots.mm)
plot!(1:update:num_days*24*3600,  ILC_power_update_mean_node_pd[1:num_days*n_updates_per_day], label=latexstring("\$u_$node^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
				 xtickfontsize=18,
				 legendfontsize=10, linewidth=3,xaxis=("days [c]",font(14)),yaxis=("normed power",font(14)),legend=false, lc =:orchid, margin=5Plots.mm)

#plot!(1:update:num_days*24*3600,  ILC_power_update_mean_node[1:num_days*n_updates_per_day], label=latexstring("\$u_{p}^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
#			   xtickfontsize=18,w=2,
#			   legendfontsize=10 ,linewidth=3, yaxis=("normed power",font(14)),xaxis=("days [c]",font(14)), seriescolor =:black, margin=5Plots.mm)
#plot!(1:update:num_days*24*3600,  ILC_power_update_mean_node[1:num_days*n_updates_per_day], label=latexstring("\$u_{p}^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
#			   xtickfontsize=18,w=2,yticks=false,
#			   legendfontsize=10 ,linewidth=3,  seriescolor =:black, margin=5Plots.mm)
plot!(1:update:num_days*24*3600,  ILC_power_update_mean_node[1:num_days*n_updates_per_day], label=latexstring("\$u_$node^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
			   	xtickfontsize=18,
			   	legendfontsize=10, linewidth=3,xaxis=("days [c]",font(14)),yaxis=("normed power",font(14)),legend=false, lc =:black, margin=5Plots.mm)


ylims!(-0.7,1.5)
title!(L"j = 3")
savefig("$dir/plots/demand_seconds_Y$(coupfact)_node_$(node)_hetero_update_half_hour.png")

node = 4
p4 = Plots.plot()
ILC_power_hourly_mean_node = vcat(ILC_power[:,:,node]'...)
ILC_power_update_mean_node_pd = vcat(ILC_power_pd[:,:,node]'...)

dd = t->((periodic_demand(t) .+ residual_demand(t)))
Plots.plot(0:num_days*l_day, t -> dd(t)[node], alpha=0.2,legend=false, label = latexstring("P^d"),linewidth=3,legendfontsize=10, linestyle=:dot)

plot!(1:update:24*num_days*3600,update_energy_pd[1:num_days*n_updates_per_day,node]./update, label=latexstring("y_{pd}"),linewidth=0.3, color ="yellow" ,  w = 2,legend=false,legendfontsize=10,legendfontrotation=10) #, linestyle=:dash)
plot!(1:update:24*num_days*3600,update_energy[1:num_days*n_updates_per_day,node]./update, label=latexstring("y_{p}"),linewidth=0.3,color="red",linestyle=:solid, w = 2)

plot!(1:update:num_days*24*3600,  ILC_power_update_mean_node_pd[1:num_days*n_updates_per_day], label=latexstring("\$u_{pd}^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)) ,ytickfontsize=14,
			   	xtickfontsize=18,
			   	legendfontsize=10,yticks=false, linewidth=3,xaxis=("days [c]",font(14)),w=2,   seriescolor = :orchid,  margin=5Plots.mm)

plot!(1:update:num_days*24*3600,  ILC_power_update_mean_node[1:num_days*n_updates_per_day], label=latexstring("\$u_{p}^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
			   xtickfontsize=18,w=2,yticks=false,
			   legendfontsize=10 ,linewidth=3,xaxis=("days [c]",font(14)),seriescolor =:black, margin=5Plots.mm)



ylims!(-0.7,1.5)
title!(L"j = 4")
savefig("$dir/plots/demand_seconds_Y$(coupfact)_node_$(node)_hetero_update_half_hour.png")

l = @layout [a b; c d]
plot_demand = Plots.plot(p1,p2,p3,p4,layout = l)
savefig(plot_demand, "$dir/plots/demand_seconds_Y$(coupfact)_all_nodes_hetero.png")

l2 = @layout [a b]
plot_demand2 = Plots.plot(p2,p4,layout = l2)
savefig(plot_demand2, "$dir/plots/demand_seconds_Y$(coupfact)_nodes2+4_hetero_update_half_hour.png")

psum = Plots.plot()
ILC_power_update_mean_sum = vcat(ILC_power[:,:,1]'...) .+ vcat(ILC_power[:,:,2]'...) .+ vcat(ILC_power[:,:,3]'...) .+ vcat(ILC_power[:,:,4]'...)
ILC_power_update_mean_sum_pd = vcat(ILC_power_pd[:,:,1]'...) .+ vcat(ILC_power_pd[:,:,2]'...) .+ vcat(ILC_power_pd[:,:,3]'...) .+ vcat(ILC_power_pd[:,:,4]'...)

dd = t->((periodic_demand(t) .+ residual_demand(t)))
plot!(0:num_days*l_day, t -> (dd(t)[1] .+ dd(t)[2] .+ dd(t)[3] .+ dd(t)[4]), alpha=0.2, label = latexstring("P^d"),linewidth=3,legendfontsize=10, linestyle=:dot)

plot!(1:update:24*num_days*3600,(update_energy_pd[1:num_days*n_updates_per_day,1] + update_energy_pd[1:num_days*n_updates_per_day,2] + update_energy_pd[1:num_days*n_updates_per_day,3] + update_energy_pd[1:num_days*n_updates_per_day,4])./update, label=latexstring("y_{pd}"),linewidth=0.3, color ="yellow" ,  w = 2,legend=false,legendfontsize=10,legendfontrotation=10)
plot!(1:update:24*num_days*3600,(update_energy[1:num_days*n_updates_per_day,1] + update_energy[1:num_days*n_updates_per_day,2] + update_energy[1:num_days*n_updates_per_day,3] + update_energy[1:num_days*n_updates_per_day,4])./update, label=latexstring("y_{p}"),linewidth=0.3,color="red",linestyle=:solid, w = 2)


plot!(1:update:num_days*24*3600,  ILC_power_update_mean_sum_pd[1:num_days*n_updates_per_day], label=latexstring("\$u_j^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
				xtickfontsize=14,
				legendfontsize=10, linewidth=3, yaxis=("normed power",font(14)),  seriescolor = :orchid,  margin=5Plots.mm)

plot!(1:update:num_days*24*3600,  ILC_power_update_mean_sum[1:num_days*n_updates_per_day], label=latexstring("\$u_j^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
				xtickfontsize=14,w=2,
				legendfontsize=10 ,linewidth=3, yaxis=("normed power",font(14)), seriescolor =:black, margin=5Plots.mm)


#ylims!(-2,3)
title!("ILC_power_update_mean_sum")
savefig(psum,"$dir/plots/demand_seconds_Y$(coupfact)_sum_hetero_update_half_hour.png")



# hourly plotting
using LaTeXStrings
Plots.plot(0:(num_days)*n_updates_per_day-1, ILC_power_update[1:num_days*n_updates_per_day], label=L"$\max_h \Vert P_{ILC, k}\Vert$", xticks = (1:n_updates_per_day:n_updates_per_day*num_days, string.(1:num_days)))
Plots.plot(0:(num_days)*n_updates_per_day-1, ILC_power_update_pd[1:num_days*n_updates_per_day], label=L"$\max_h \Vert P_{ILC, k}\Vert$", xticks = (1:n_updates_per_day:n_updates_per_day*num_days, string.(1:num_days)))


plot!(0:n_updates_per_day*num_days-1,norm_update_energy./update, label=L"y_h")
plot!(0:n_updates_per_day*num_days-1,norm_update_energy_pd./update, label=L"y_h")

plot!(0:n_updates_per_day:n_updates_per_day*num_days-1, load_amp, label = "demand amplitude")

xlabel!("hour h [h]")
ylabel!("normed quantities [a.u.]")
savefig("$dir/plots/yh_demand_ILC_new_hourly_hetero_update_half_hour.png")



Plots.plot()
# daily plotting
Plots.plot(1:num_days, ILC_power_agg[1:num_days,1,1] ./ maximum(ILC_power_agg), label=L"$\max_h \Vert P_{ILC, k}\red$")
Plots.plot!(1:num_days, ILC_power_agg_pd[1:num_days,1,1] ./ maximum(ILC_power_agg_pd), label=L"$\max_h \Vert P_{ILC, k}\Vert$")

plot!(1:num_days, mean(norm_energy_d,dims=2) ./ maximum(norm_energy_d), label=L"norm(y_h)")
plot!(1:num_days, mean(norm_energy_d_pd,dims=2) ./ maximum(norm_energy_d_pd), label=L"norm(y_h)")

plot!(1:num_days, load_amp  ./ maximum(load_amp), label = "demand amplitude")
xlabel!("day d [d]")
ylabel!("normed quantities [a.u.]")
savefig("$dir/plots/demand_daily_hetero_update_half_hour.png")
