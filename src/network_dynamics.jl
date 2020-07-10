@doc """
This is a module that contains the system and control dynamics.
# Examples
```julia-repl
julia> include("src/network_dynamics.jl")
Main.network_dynamics
```
"""
module network_dynamics

begin
	using Random # random numbers
	using LightGraphs # create network topologies
	using LinearAlgebra
	using DifferentialEquations: reinit!
	using DSP
	using ToeplitzMatrices
end

l_hour = 3600 # DemCurve.update
l_day = 3600 * 24
# update = l_hour/4  #/2 for half # DemCurve.update
# n_updates_per_day=96 #l_day / update

@doc """
    ACtoymodel!(du, u, p, t)
Lower-layer dynamics with controller from [Dörfler et al. 2017] eqns. 15a,b,c.
with kp = D, kI = K and chi = -p
[Dörfler et al. 2017]: https://arxiv.org/pdf/1711.07332.pdf
"""


function ACtoymodel!(du, u, p, t)
	theta = view(u, 1:p.N)
	omega = view(u, (p.N+1):(2*p.N))
	chi = view(u, (2*p.N+1):(3*p.N))

	dtheta = view(du, 1:p.N)
	domega = view(du, (p.N+1):(2*p.N))
	dchi = view(du, (2*p.N+1):(3*p.N))

	control_power_integrator = view(du,(3*p.N+1):(4*p.N))
	control_power_integrator_abs = view(du,(4*p.N+1):(5*p.N))

	# demand = - p.periodic_demand(t) .- p.residual_demand(t)
	power_ILC = p.hl.current_background_power #(t)
	power_LI =  chi .- p.ll.kP .* omega
	periodic_power = - p.periodic_demand(t) .+ p.periodic_infeed(t)
	fluctuating_power = - p.residual_demand(t) .+ p.fluctuating_infeed(t) # here we can add fluctuating infeed as well
	# avoid *, use mul! instead with pre-allocated cache http://docs.juliadiffeq.org/latest/basics/faq.html
	#cache1 = zeros(size(p.coupling)[1])
	#cache2 = similar(cache1)
	#flows = similar(theta)
	#mul!(cache1, p.incidence', theta)
	#mul!(cache2, p.coupling, sin.(cache1))
	#mul!(flows, - p.incidence, cache2 )
	flows = - (p.incidence * p.coupling * sin.(p.incidence' * theta))


	@. dtheta = omega
	@. domega = p.ll.M_inv .* (power_ILC .+ power_LI
						.+ periodic_power .+ fluctuating_power .+ flows)
						# signs checked (Ruth)
    @. dchi = p.ll.T_inv .* (- omega .- p.ll.kI .* chi) # Integrate the control power used.
	@. control_power_integrator = power_LI
	@. control_power_integrator_abs = abs.(power_LI)
	return nothing
end

@doc """
    ACtoymodel!(du, u, p, t)
Lower-layer dynamics with controller from [Dörfler et al. 2017] eqns. 15a,b,c.
with kp = D, kI = K and chi = -p
[Dörfler et al. 2017]: https://arxiv.org/pdf/1711.07332.pdf
with DC approx (sin removed)
"""

function ACtoymodel_lin!(du, u, p, t)
	theta = view(u, 1:p.N)
	omega = view(u, (p.N+1):(2*p.N))
	chi = view(u, (2*p.N+1):(3*p.N))

	dtheta = view(du, 1:p.N)
	domega = view(du, (p.N+1):(2*p.N))
	dchi = view(du, (2*p.N+1):(3*p.N))

	control_power_integrator = view(du,(3*p.N+1):(4*p.N))
	control_power_integrator_abs = view(du,(4*p.N+1):(5*p.N))

	# demand = - p.periodic_demand(t) .- p.residual_demand(t)
	power_ILC = p.hl.current_background_power
	power_LI =  chi .- p.ll.kP .* omega
	periodic_power = - p.periodic_demand(t) .+ p.periodic_infeed(t)
	fluctuating_power = - p.residual_demand(t) .+ p.fluctuating_infeed(t) # here we can add fluctuating infeed as well
	# avoid *, use mul! instead with pre-allocated cache http://docs.juliadiffeq.org/latest/basics/faq.html
	#cache1 = zeros(size(p.coupling)[1])
	#cache2 = similar(cache1)
	#flows = similar(theta)
	#mul!(cache1, p.incidence', theta)
	#mul!(cache2, p.coupling, sin.(cache1))
	#mul!(flows, - p.incidence, cache2 )
	flows = - (p.incidence * p.coupling * p.incidence' * theta)


	@. dtheta = omega
	@. domega = p.ll.M_inv .* (power_ILC .+ power_LI
						.+ periodic_power .+ fluctuating_power .+ flows)
						# signs checked (Ruth)
    @. dchi = p.ll.T_inv .* (- omega .- p.ll.kI .* chi) # Integrate the control power used.
	@. control_power_integrator = power_LI
	@. control_power_integrator_abs = abs.(power_LI)
	return nothing
end


@doc """
    Updating()
Store the integrated control power in memory.
See also [`(hu::Updating)`](@ref).
"""
struct Updating
	integrated_control_power_history
	Updating() = new([])
end



@doc """
    Updating(integrator)
PeriodicCallback function acting on the `integrator` that is called every simulation hour (t = 1,2,3...).
"""
function (hu::Updating)(integrator)
	#integrator.p.hl.update=update_lst[batch]

	n_updates_per_day = Int(floor(l_day/integrator.p.hl.update))
	updating_cycle  = Int(floor(mod(round(Int, integrator.t/integrator.p.hl.update), n_updates_per_day) + 1))
	last_update = Int(floor(mod(updating_cycle-2, n_updates_per_day) + 1))

	power_idx = 3*integrator.p.N+1:4*integrator.p.N
	power_abs_idx = 4*integrator.p.N+1:5*integrator.p.N
	# For the array  of arrays append to work correctly we need to give append!
	# an array of arrays. Otherwise obscure errors follow. Therefore u[3...] is
	# wrapped in [].

	# append!(hu.integrated_control_power_low_layer_controlhistory, [integrator.u[power_idx]])

	# println("===========================")
	# println("Starting hour $hour, last hour was $last_update")
	# println("Integrated power from the last hour:")
	# println(integrator.u[power_idx])
	# println("Yesterdays mismatch for the last hour:")
	# println(integrator.p.hl.mismatch_yesterday[last_update,:])
	# println("Background power for the next hour:")
	# println(integrator.p.hl.daily_background_power[hour, :])
	integrator.p.hl.mismatch_yesterday[last_update,:] .= integrator.u[power_idx]
	integrator.p.hl.mismatch_d_control[last_update,:] .= integrator.p.hl.mismatch_yesterday[last_update,:]
	integrator.u[power_idx] .= 0.
	integrator.u[power_abs_idx] .= 0.

	# println("hour $hour")
	integrator.p.hl.current_background_power .= integrator.p.hl.daily_background_power[updating_cycle, :]
	# integrator.p.residual_demand = 0.1 * (0.5 + rand()
	# reinit!(integrator, integrator.u, t0=integrator.t, erase_sol=true)

	#now = copy(integrator.t)
	#state = copy(integrator.u)
	#reinit!(integrator, state, t0=now, erase_sol=true)
	nothing
end


# function (hu::Updating_d)(integrator)
# 	#integrator.p.hl.update=update_lst[batch]
#
# 	n_updates_per_day = Int(floor(l_day/integrator.p.hl.update))
# 	integrator.t
# 	updating_cycle  = Int(floor(mod(round(Int, integrator.t/integrator.p.hl.update), n_updates_per_day) + 1))
# 	last_update = Int(floor(mod(updating_cycle-2, n_updates_per_day) + 1))
#
# 	power_idx = 3*integrator.p.N+1:4*integrator.p.N
# 	power_abs_idx = 4*integrator.p.N+1:5*integrator.p.N
#
# 	integrator.p.hl.mismatch_yesterday[last_update,last_iteration] .= integrator.u[power_idx]
# 	integrator.u[power_idx] .= 0.
# 	integrator.u[power_abs_idx] .= 0.
#
# 	# println("hour $hour")
# 	integrator.p.hl.current_background_power .= integrator.p.hl.daily_background_power[updating_cycle, :]
# 	# integrator.p.residual_demand = 0.1 * (0.5 + rand()
# 	# reinit!(integrator, integrator.u, t0=integrator.t, erase_sol=true)
#
# 	#now = copy(integrator.t)
# 	#state = copy(integrator.u)
# 	#reinit!(integrator, state, t0=now, erase_sol=true)
# 	nothing
# end


function DailyUpdate_X(integrator)
	#println("mismatch ", integrator.p.hl.daily_background_power)
	#println("Q ", integrator.p.hl.Q)
	integrator.p.hl.daily_background_power = integrator.p.hl.Q * (integrator.p.hl.daily_background_power + integrator.p.hl.kappa * integrator.p.hl.mismatch_yesterday)
	#println("mismatch ", integrator.p.hl.daily_background_power)
	nothing
end

function DailyUpdate_PD(integrator)
	#println("mismatch ", integrator.p.hl.daily_background_power)
	#println("Q ", integrator.p.hl.Q)

	integrator.p.hl.daily_background_power = integrator.p.hl.Q * ( integrator.p.hl.daily_background_power + integrator.p.hl.kappa * integrator.p.hl.mismatch_yesterday
	+integrator.p.hl.kappa *(integrator.p.hl.mismatch_yesterday - integrator.p.hl.mismatch_d_control) )

	#println("mismatch ", integrator.p.hl.daily_background_power)

	#end
	nothing
end

end
