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
	# low_layer_control = system_structs.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner=N),kI=repeat([0.005], inner=N)) # different for each node, change array
	low_layer_control = system_structs.LeakyIntegratorPars(M_inv=[1/5.; 1/4.8; 1/4.1; 1/4.8],kP= [400.; 110.; 100.; 200.],T_inv=[1/0.04; 1/0.045; 1/0.047; 1/0.043],kI=[0.05; 0.004; 0.05; 0.001]) # different for each node, change array
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=[0.1; 10; 100; 1000],T_inv=repeat([1/0.05], inner=N),kI=repeat([0.005], inner=N)) # different for each node, change array
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=[1/0.05; 1/0.5; 1/5; 1/50],kI=repeat([0.005], inner=N)) # different for each node, change array
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=repeat([0.2], inner=N),kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner = N),kI=[0.005; 0.5; 5; 500]) # different for each node, change array
	#low_layer_control = system_structs.LeakyIntegratorPars(M_inv=[0.002; 0.2; 2; 20],kP=repeat([525], inner=N),T_inv=repeat([1/0.05], inner = N),kI=repeat([0.005], inner=N)) # different for each node, change array
	kappa =(1.0 / update) #*2 for update half
	empty = false
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
Q = Toeplitz(Q1[1001:1001+n_updates_per_day-1],Q1[1001:1001+n_updates_per_day-1]);
#Q = Toeplitz(Q1[1001:1001+24-1],Q1[1001:1001+24-1]);


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

#using Pandas
using DataFrames, StatsPlots , StringEncodings , DataFramesMeta

begin
	factor = 0#0.01*rand(compound_pars.D * compound_pars.N)#0.001#0.00001
	ic = factor .* ones(compound_pars.D * compound_pars.N)
	tspan = (0., num_days * l_day)
	ode_tl1 = ODEProblem(network_dynamics.ACtoymodel!, ic, tspan, compound_pars,
	callback=CallbackSet(PeriodicCallback(network_dynamics.Updating(),update),
						 PeriodicCallback(network_dynamics.DailyUpdate_X, l_day)))
end
sol1 = solve(ode_tl1, Rodas4())


df=DataFrame(sol1)

df_t=DataFrame(t=sol1.t)
# df_sol = DataFrame(sol1')
CSV.write("pd2u.csv",df)
CSV.write("pd2t.csv",df_t)


# CSV.write("pd_test.csv",df_sol)
# daten_test = CSV.read("pd_test.csv")

daten_u = CSV.read("pd2u.csv" )
#tab=[frames.Column2,frames.Column3,frames.Column4,frames.Column5,frames.Column6,frames.Column7,frames.Column8,frames.Column9,frames.Column10,frames.Column11,frames.Column12,frames.Column13,frames.Column14,frames.Column15,frames.Column16,frames.Column17,frames.Column18,frames.Column19,frames.Column20,frames.Column21]
#tab2=[frames.x1,frames.x2,frames.x3,frames.x4,frames.x5,frames.x6,frames.x7,frames.x8,frames.x9,frames.x10,frames.Column11,frames.Column12,frames.Column13,frames.Column14,frames.Column15,frames.Column16,frames.Column17,frames.Column18,frames.Column19,frames.Column20,frames.Column21]

daten_t = CSV.read("pd2t.csv"; header=false , transpose =true )
deletecols!(daten_t, :Column1)
#daten_t=daten_t[:,2:37223]
frames=hcat(daten_t,daten_u)
CSV.write("pd2.csv",frames)
dfk=DataFrame(t=frames.t , u=tab)
#CSV.File(open(read, "pd2.csv", enc"ISO-8859-1")) |> DataFrame
#daten_num = [parse(Float64, ss) for ss in split(daten.u)]


names!(daten_t, [Symbol("$i") for i in 1:size(df,2)])
names!(daten_u, [Symbol("$i") for i in 1:size(df,2)])

#df = DataFrame(sol1)
using CSV
using JLD2 , Pandas
#JLD2.@save "outputt.jld2" sol1

##################### [3.5841198212278796e-10, 3.078381669806953e-10, -7.730920518542944e-12, 3.5113574636468243e-10, 4.18391402683452e-6, 3.5993826772601687e-6, -9.03894389117365e-8, 4.10344664907011e-6, -8.959660667241752e-9, -6.840813496031148e-9, 1.6447769689790595e-10, -8.16593676270488e-9, -1.4336530395777753e-7, -3.3862588282699924e-8, 7.731014271411215e-10, -7.022761484130914e-8, 1.3708213968165875e-7, 3.2382926029415346e-8, 7.731014271411215e-10, 6.715593233297654e-8]##################################################
#                               PLOTTING                             #
###################''''''''''''''''''''''''###################################################
using Plots

update_energy = zeros(n_updates_per_day*num_days+1,N)
for i=1:n_updates_per_day*num_days+1
	for j = 1:N
		update_energy[i,j] = sol1((i-1)*update)[energy_filter[j]]
	end
end



update_energy_pd2 = zeros(n_updates_per_day*num_days+1,N)
for i=1:n_updates_per_day*num_days+1
	for j = 1:N
		for k in 1:size(daten_t,2)
				  if (daten_t[k][1] ==(i-1)*update)
					update_energy_pd2[i,j] =   daten_u[k][energy_filter[j]]
				  end
		 end
	end
end

empty = false
for k in 1:size(daten_t,2)
		   if ((daten_t[k][1] ==(2-1)*update)&& (empty == true ))
				   @show daten_u[k][energy_filter[1]]
				   empty= false
		   else
			   	   empty = true
		   end
 end

plot(update_energy)
plot!(update_energy_pd2)
using Images
#img = load("$dir/plots/demand_seconds_hetero_update_energy.png")
#plot(load("$dir/plots/demand_seconds_hetero_update_energy.png"))
#p1 = plot()


#plot!(update_energy[:,1] , title = "Update energy")
#plot!(convert(Matrix, data[:,1:4])[:,1] )
#@df df plot(:update_energy)
savefig("$dir/plots/demand_seconds_hetero_update_energy.png")
#hdf5() #Select HDF5-Plots "backend"
#p = plot(update_energy[:,1] , title = "Update energy") #Construct plot as usual#
#Plots.hdf5plot_write(p, "plotsave.hdf5")
#plot() #Must first select some backend
#pread = Plots.hdf5plot_read("plotsave.hdf5")
#plot!(update_energy[:,1] , title = "Update energy")
#savefig("hour.png")

#Then, write to .hdf5 file:

#After you re-open a new Julia session, you can re-read the .hdf5 plot:


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
		norm_energy_d[i,j] = norm(update_energy[(i-1)*n_updates_per_day+1:i*n_updates_per_day,j])
	end
end



ILC_power_pd2 = zeros(num_days+2,n_updates_per_day,N)
for j = 1:N
	ILC_power_pd2[2,:,j] = Q*(zeros(n_updates_per_day,1) +  kappa*update_energy_pd2[1:n_updates_per_day,j])
end
norm_energy_d_pd2 = zeros(num_days,N)
for j = 1:N
	norm_energy_d_pd2[1,j] = norm(update_energy_pd2[1:n_updates_per_day,j])
end


for i=2:num_days
	for j = 1:N
		ILC_power_pd2[i+1,:,j] = Q*(ILC_power_pd2[i,:,j] +  kappa*update_energy_pd2[(i-1)*n_updates_per_day+1:i*n_updates_per_day,j])
		norm_energy_d_pd2[i,j] = norm(update_energy_pd2[(i-1)*n_updates_per_day+1:i*n_updates_per_day,j])
	end
end



#ILC_power_agg = maximum(mean(ILC_power.^2,dims=3),dims=2)
ILC_power_agg = [norm(mean(ILC_power,dims=3)[d,:]) for d in 1:num_days+2]
ILC_power_update_mean = vcat(mean(ILC_power,dims=3)[:,:,1]'...)
ILC_power_update_mean_node1 = vcat(ILC_power[:,:,1]'...)
ILC_power_update = [norm(reshape(ILC_power,(num_days+2)*n_updates_per_day,N)[h,:]) for h in 1:n_updates_per_day*(num_days+2)]
ILC_power_update_node1 = [norm(reshape(ILC_power,(num_days+2)*n_updates_per_day,N)[h,1]) for h in 1:n_updates_per_day*(num_days+2)]
dd = t->((periodic_demand(t) .+ residual_demand(t))./100)
load_amp = [first(maximum(dd(t))) for t in 1:3600*24:3600*24*num_days]
norm_update_energy = [norm(update_energy[h,:]) for h in 1:n_updates_per_day*num_days]

ILC_power_agg_pd2 = [norm(mean(ILC_power_pd2,dims=3)[d,:]) for d in 1:num_days+2]
ILC_power_update_mean_pd2 = vcat(mean(ILC_power_pd2,dims=3)[:,:,1]'...)
ILC_power_update_mean_node1_pd2 = vcat(ILC_power_pd2[:,:,1]'...)
ILC_power_update_pd2 = [norm(reshape(ILC_power_pd2,(num_days+2)*n_updates_per_day,N)[h,:]) for h in 1:n_updates_per_day*(num_days+2)]
ILC_power_update_node1_pd2 = [norm(reshape(ILC_power_pd2,(num_days+2)*n_updates_per_day,N)[h,1]) for h in 1:n_updates_per_day*(num_days+2)]
dd = t->((periodic_demand(t) .+ residual_demand(t))./100)
load_amp = [first(maximum(dd(t))) for t in 1:3600*24:3600*24*num_days]
norm_update_energy_pd2 = [norm(update_energy_pd2[h,:]) for h in 1:n_updates_per_day*num_days]

using LaTeXStrings
#df = DataFrame(kappa_pd2 = [p[6] for p in sol1.u] ,
#			   update_energy_pd2 = [p[10] for p in sol1.u],
#			   norm_energy_d_pd2 = [p[11] for p in sol1.u],
#			   update_pd2 = [p[12] for p in sol1.u])

# NODE WISE second-wisenode = 1
node = 1
p1 = plot()
ILC_power_update_mean_node = vcat(ILC_power[:,:,node]'...)

ILC_power_update_mean_node_pd2 = vcat(ILC_power_pd2[:,:,node]'...)
dd = t->((periodic_demand(t) .+ residual_demand(t)))
plot!(0:num_days*l_day, t -> dd(t)[node], alpha=0.2, label = latexstring("P^d_$node"),linewidth=3, linestyle=:dot)


plot!(1:update:24*num_days*3600,update_energy[1:num_days*n_updates_per_day,node]./update, label=latexstring("y_$node^{c,h}"),linewidth=2,linestyle = :dot) #, linestyle=:dash)

plot!(1:update:24*num_days*3600,update_energy_pd2[1:num_days*n_updates_per_day,node]./update, label=latexstring("y_$node^{c,h}"),linewidth=2,linestyle = :dot, lc =:black) #, linestyle=:dash)


plot!(1:update:num_days*24*3600,  ILC_power_update_mean_node[1:num_days*n_updates_per_day], label=latexstring("\$u_$node^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
               xtickfontsize=14,
    		   legendfontsize=10, linewidth=3, yaxis=("normed power",font(14)),legend=false, lc =:black, margin=5Plots.mm)

plot!(1:update:num_days*24*3600,  ILC_power_update_mean_node_pd2[1:num_days*n_updates_per_day], label=latexstring("\$u_$node^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
			   xtickfontsize=14,
			   legendfontsize=10, linewidth=3, yaxis=("normed power",font(14)),legend=false, lc =:red,  markerstrokestyle = :dot, margin=5Plots.mm)

ylims!(-0.7,1.5)
title!(L"j = 1")


#df = DataFrame(update_energy)
#CSV.write("ILC_power_update_mean_node.csv",df)
#ILC_power_update_mean_node_csv = CSV.read("ILC_power_update_mean_node.csv")
#plot(ILC_power_update_mean_node[:,1] , title = "Update energy")
#plot!(1:update:num_days*24*3600,  ILC_power_update_mean_node[1:num_days*n_updates_per_day], label=latexstring("\$u_$node^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
#               xtickfontsize=14,
#    		   legendfontsize=10, linewidth=3, yaxis=("normed power",font(14)),legend=false, lc =:black, margin=5Plots.mm)


savefig("$dir/plots/demand_seconds_Y$(coupfact)_node_$(node)_hetero_update_half_hour.png")

node = 2
p2 = plot()
ILC_power_update_mean_node = vcat(ILC_power[:,:,node]'...)
ILC_power_update_mean_node_pd2 = vcat(ILC_power_pd2[:,:,node]'...)


dd = t->((periodic_demand(t) .+ residual_demand(t)))
plot!(0:num_days*l_day, t -> dd(t)[node], alpha=0.2, label = latexstring("P^d_$node"),linewidth=3, linestyle=:dot)

plot!(1:update:24*num_days*3600,update_energy[1:num_days*n_updates_per_day,node]./update, label=latexstring("y_$node^{c,h}"),linewidth=3)#, linestyle=:dash)
plot!(1:update:24*num_days*3600,update_energy_pd2[1:num_days*n_updates_per_day,node]./update, label=latexstring("y_$node^{c,h}"),linewidth=3)#, linestyle=:dash)

plot!(1:update:num_days*24*3600,  ILC_power_update_mean_node[1:num_days*n_updates_per_day], label=latexstring("\$u_$node^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
               xtickfontsize=14, yticks=false, #xaxis=("days [c]",font(14)), yaxis=("normed power",font(14))
    		   legendfontsize=10, linewidth=3,legend=false, lc =:black, margin=5Plots.mm)
plot!(1:update:num_days*24*3600,  ILC_power_update_mean_node_pd2[1:num_days*n_updates_per_day], label=latexstring("\$u_$node^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
			   xtickfontsize=14, yticks=false, #xaxis=("days [c]",font(14)), yaxis=("normed power",font(14))
			   legendfontsize=10, linewidth=3,legend=false, lc =:black, margin=5Plots.mm)

ylims!(-0.7,1.5)
title!(L"j = 2")
savefig("$dir/plots/demand_seconds_Y$(coupfact)_node_$(node)_hetero_update_half_hour.png")

node = 3
p3 = plot()
ILC_power_update_mean_node = vcat(ILC_power[:,:,node]'...)
ILC_power_update_mean_node_pd2 = vcat(ILC_power_pd2[:,:,node]'...)
dd = t->((periodic_demand(t) .+ residual_demand(t)))
plot!(0:num_days*l_day, t -> dd(t)[node], alpha=0.2, label = latexstring("P^d_$node"),linewidth=3, linestyle=:dot)

plot!(1:update:24*num_days*3600,update_energy[1:num_days*n_updates_per_day,node]./update, label=latexstring("y_$node^{c,h}"),linewidth=3)#, linestyle=:dash)
plot!(1:update:24*num_days*3600,update_energy_pd2[1:num_days*n_updates_per_day,node]./update, label=latexstring("y_$node^{c,h}"),linewidth=3)#, linestyle=:dash)

plot!(1:update:num_days*24*3600,  ILC_power_update_mean_node[1:num_days*n_updates_per_day], label=latexstring("\$u_$node^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
               xtickfontsize=18,
    		   legendfontsize=10, linewidth=3,xaxis=("days [c]",font(14)),yaxis=("normed power",font(14)),legend=false, lc =:black, margin=5Plots.mm)
 plot!(1:update:num_days*24*3600,  ILC_power_update_mean_node_pd2[1:num_days*n_updates_per_day], label=latexstring("\$u_$node^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
			    xtickfontsize=18,
			    legendfontsize=10, linewidth=3,xaxis=("days [c]",font(14)),yaxis=("normed power",font(14)),legend=false, lc =:black, margin=5Plots.mm)

ylims!(-0.7,1.5)
title!(L"j = 3")
savefig("$dir/plots/demand_seconds_Y$(coupfact)_node_$(node)_hetero_update_half_hour.png")

node = 4
p4 = plot()
ILC_power_update_mean_node = vcat(ILC_power[:,:,node]'...)
ILC_power_update_mean_node_pd2 = vcat(ILC_power_pd2[:,:,node]'...)

dd = t->((periodic_demand(t) .+ residual_demand(t)))

plot!(0:num_days*l_day, t -> dd(t)[node], alpha=0.2, label = latexstring("P^d_$node"),linewidth=3, linestyle=:dot)
plot!(1:update:24*num_days*3600,update_energy[1:num_days*n_updates_per_day,node]./update, label=latexstring("y_$node^{c,h}"),linewidth=3)#, linestyle=:dash)
plot!(1:update:24*num_days*3600,update_energy_pd2[1:num_days*n_updates_per_day,node]./update, label=latexstring("y_$node^{c,h}"),linewidth=3)#, linestyle=:dash)

plot!(1:update:num_days*24*3600,  ILC_power_update_mean_node[1:num_days*n_updates_per_day], label=latexstring("\$u_$node^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
               xtickfontsize=18,
    		   legendfontsize=10, yticks=false, linewidth=3,xaxis=("days [c]",font(14)),legend=false, lc =:black, margin=5Plots.mm)
plot!(1:update:num_days*24*3600,  ILC_power_update_mean_node_pd2[1:num_days*n_updates_per_day], label=latexstring("\$u_$node^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
			   xtickfontsize=18,
			   legendfontsize=10, yticks=false, linewidth=3,xaxis=("days [c]",font(14)),legend=false, lc =:black, margin=5Plots.mm)

ylims!(-0.7,1.5)
title!(L"j = 4")
savefig("$dir/plots/demand_seconds_Y$(coupfact)_node_$(node)_hetero_update_half_hour.png")

l = @layout [a b; c d]
plot_demand = plot(p1,p2,p3,p4,layout = l)
savefig(plot_demand, "$dir/plots/demand_seconds_Y$(coupfact)_all_nodes_hetero_update_half_hour.png")

l2 = @layout [a b]
plot_demand2 = plot(p2,p4,layout = l2)
savefig(plot_demand2, "$dir/plots/demand_seconds_Y$(coupfact)_nodes2+4_hetero_update_half_hour.png")

psum = plot()
ILC_power_update_mean_sum = vcat(ILC_power[:,:,1]'...) .+ vcat(ILC_power[:,:,2]'...) .+ vcat(ILC_power[:,:,3]'...) .+ vcat(ILC_power[:,:,4]'...)
ILC_power_update_mean_sum_pd2 = vcat(ILC_power_pd2[:,:,1]'...) .+ vcat(ILC_power_pd2[:,:,2]'...) .+ vcat(ILC_power_pd2[:,:,3]'...) .+ vcat(ILC_power_pd2[:,:,4]'...)

dd = t->((periodic_demand(t) .+ residual_demand(t)))
plot!(0:num_days*l_day, t -> (dd(t)[1] .+ dd(t)[2] .+ dd(t)[3] .+ dd(t)[4]), alpha=0.2, label = latexstring("\$P^d_j\$"),linewidth=3, linestyle=:dot)

plot!(1:update:24*num_days*3600,(update_energy[1:num_days*n_updates_per_day,1] + update_energy[1:num_days*n_updates_per_day,2] + update_energy[1:num_days*n_updates_per_day,3] + update_energy[1:num_days*n_updates_per_day,4])./update, label=latexstring("y_j^{c,h}"),linewidth=3, linestyle=:dash)
plot!(1:update:24*num_days*3600,(update_energy_pd2[1:num_days*n_updates_per_day,1] + update_energy_pd2[1:num_days*n_updates_per_day,2] + update_energy_pd2[1:num_days*n_updates_per_day,3] + update_energy_pd2[1:num_days*n_updates_per_day,4])./update, label=latexstring("y_j^{c,h}"),linewidth=3, linestyle=:dash)


plot!(1:update:num_days*24*3600,  ILC_power_update_mean_sum[1:num_days*n_updates_per_day], label=latexstring("\$u_j^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
               xtickfontsize=18,legend=false,
    		   legendfontsize=10, linewidth=3,xaxis=("days [c]",font(14)),  yaxis=("normed power",font(14)),lc =:black, margin=5Plots.mm)
plot!(1:update:num_days*24*3600,  ILC_power_update_mean_sum_pd2[1:num_days*n_updates_per_day], label=latexstring("\$u_j^{ILC}\$"), xticks = (0:3600*24:num_days*24*3600, string.(0:num_days)), ytickfontsize=14,
			    xtickfontsize=18,legend=false,
			    legendfontsize=10, linewidth=3,xaxis=("days [c]",font(14)),  yaxis=("normed power",font(14)),lc =:black, margin=5Plots.mm)
#ylims!(-0.7,1.5)
title!("ILC_power_update_mean_sum")
savefig(psum,"$dir/plots/demand_seconds_Y$(coupfact)_sum_hetero_update_half_hour.png")



# hourly plotting
using LaTeXStrings
plot(0:(num_days)*n_updates_per_day-1, ILC_power_update[1:num_days*n_updates_per_day], label=L"$\max_h \Vert P_{ILC, k}\Vert$", xticks = (1:n_updates_per_day:n_updates_per_day*num_days, string.(1:num_days)))
plot(0:(num_days)*n_updates_per_day-1, ILC_power_update_pd2[1:num_days*n_updates_per_day], label=L"$\max_h \Vert P_{ILC, k}\Vert$", xticks = (1:n_updates_per_day:n_updates_per_day*num_days, string.(1:num_days)))


plot!(0:n_updates_per_day*num_days-1,norm_update_energy./update, label=L"y_h")
plot!(0:n_updates_per_day*num_days-1,norm_update_energy_pd2./update, label=L"y_h")

plot!(0:n_updates_per_day:n_updates_per_day*num_days-1, load_amp, label = "demand amplitude")

xlabel!("hour h [h]")
ylabel!("normed quantities [a.u.]")
savefig("$dir/plots/yh_demand_ILC_new_hourly_hetero_update_half_hour.png")




# daily plotting
plot(1:num_days, ILC_power_agg[1:num_days,1,1] ./ maximum(ILC_power_agg), label=L"$\max_h \Vert P_{ILC, k}\Vert$")
plot(1:num_days, ILC_power_agg_pd2[1:num_days,1,1] ./ maximum(ILC_power_agg_pd2), label=L"$\max_h \Vert P_{ILC, k}\Vert$")

plot!(1:num_days, mean(norm_energy_d,dims=2) ./ maximum(norm_energy_d), label=L"norm(y_h)")
plot!(1:num_days, mean(norm_energy_d_pd2,dims=2) ./ maximum(norm_energy_d_pd2), label=L"norm(y_h)")

plot!(1:num_days, load_amp  ./ maximum(load_amp), label = "demand amplitude")
xlabel!("day d [d]")
ylabel!("normed quantities [a.u.]")
savefig("$dir/plots/demand_daily_hetero_update_half_hour.png")
