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
solve AC power flow
"""
function rootfunc!(du, u, p, t)
	power_ILC = p.hl.current_background_power
	periodic_power = - p.periodic_demand(t) .+ p.periodic_infeed(t)
	fluctuating_power = - p.residual_demand(t) .+ p.fluctuating_infeed(t) # here we can add fluctuating infeed as well
	flows = - (p.incidence * p.coupling * sin.(p.incidence' * u))


	@. du = p.ll.M_inv .* (power_ILC .+ periodic_power .+ fluctuating_power .+ flows)
	return nothing
end





@doc """
    HourlyUpdate()
Store the integrated control power in memory.
See also [`(hu::HourlyUpdate)`](@ref).
"""
struct HourlyUpdate
	integrated_control_power_history
	HourlyUpdate() = new([])
end



@doc """
    HourlyUpdate(integrator)
PeriodicCallback function acting on the `integrator` that is called every simulation hour (t = 1,2,3...).
"""
function (hu::HourlyUpdate)(integrator)
	hour = mod(round(Int, integrator.t/3600.), 24) + 1
	last_hour = mod(hour-2, 24) + 1

	power_idx = 3*integrator.p.N+1:4*integrator.p.N
	power_abs_idx = 4*integrator.p.N+1:5*integrator.p.N
	# For the array  of arrays append to work correctly we need to give append!
	# an array of arrays. Otherwise obscure errors follow. Therefore u[3...] is
	# wrapped in [].

	# append!(hu.integrated_control_power_history, [integrator.u[power_idx]])

	# println("===========================")
	# println("Starting hour $hour, last hour was $last_hour")
	# println("Integrated power from the last hour:")
	# println(integrator.u[power_idx])
	# println("Yesterdays mismatch for the last hour:")
	# println(integrator.p.hl.mismatch_yesterday[last_hour,:])
	# println("Background power for the next hour:")
	# println(integrator.p.hl.daily_background_power[hour, :])

	integrator.p.hl.mismatch_yesterday[last_hour,:] .= integrator.u[power_idx]
	integrator.u[power_idx] .= 0.
	integrator.u[power_abs_idx] .= 0.

	# println("hour $hour")
	integrator.p.hl.current_background_power .= integrator.p.hl.daily_background_power[hour, :]
	# integrator.p.residual_demand = 0.1 * (0.5 + rand())
	# reinit!(integrator, integrator.u, t0=integrator.t, erase_sol=true)

	#now = copy(integrator.t)
	#state = copy(integrator.u)
	#reinit!(integrator, state, t0=now, erase_sol=true)
	nothing
end


function neighbour_map(g, vc)
    Dict([v => neighbors(g, v) for v in vc])
end

function ilc_update(control_energy, g)
    # set of vertices that are independent
    # (no two vertices are adjacent to each other)
    ilc_nodes = independent_set(g, DegreeIndependentSet())
    nm = neighbour_map(g, ilc_nodes)
    return Dict([x => (control_energy[x] + sum(control_energy[nm[x]]))/length(collect(values(nm[x]))) for x in ilc_nodes])
end

function DaylyUpdate(integrator)
	@sync @. integrator.p.hl.daily_background_power += integrator.p.hl.kappa * integrator.p.hl.mismatch_yesterday
	#println("mismatch ", integrator.p.hl.daily_background_power)
	nothing
end

@doc """
    DailyUpdate_4(integrator) - vertex cover ILC with averaged update from neighbors
PeriodicCallback function acting on the `integrator` that implements the ILC once a simulation day.
"""
function DailyUpdate_4(integrator)
	#y_h = ilc_update(integrator.p.hl.mismatch_yesterday, integrator.p.graph)
	#println("y_h ", y_h)
	cover_values = [integrator.p.hl.mismatch_yesterday[:,x] + sum(integrator.p.hl.mismatch_yesterday[:,integrator.p.hl.ilc_covers[x]],dims=2)./length(collect(values(integrator.p.hl.ilc_covers[x]))) for x in integrator.p.hl.ilc_nodes]
	@sync integrator.p.hl.daily_background_power[:,integrator.p.hl.ilc_nodes] .+= integrator.p.hl.kappa .* hcat(Vector(collect(cover_values))...)
	nothing
end


@doc """
    DailyUpdate_5(integrator) - ILC at vertex cover with local update
PeriodicCallback function acting on the `integrator` that implements the ILC once a simulation day.
"""
function DailyUpdate_5(integrator)
	#cover_values = [integrator.p.hl.mismatch_yesterday[:,x] + sum(integrator.p.hl.mismatch_yesterday[:,integrator.p.hl.ilc_covers[x]],dims=2)./length(collect(values(integrator.p.hl.ilc_covers[x]))) for x in integrator.p.hl.ilc_nodes]
	@sync integrator.p.hl.daily_background_power[:,integrator.p.hl.ilc_nodes] .+= integrator.p.hl.kappa .* integrator.p.hl.mismatch_yesterday[:,integrator.p.hl.ilc_nodes]
	nothing
end

@doc """
    DaylyUpdate(integrator) - ILC at 50% of nodes and random averaged update (3 random nodes)
PeriodicCallback function acting on the `integrator` that implements the ILC once a simulation day.
"""
function DailyUpdate_6(integrator)
	@sync integrator.p.hl.daily_background_power +=  integrator.p.hl.mismatch_yesterday * integrator.p.hl.kappa'
	nothing
end

@doc """
    DaylyUpdate(integrator) - local ILC
PeriodicCallback function acting on the `integrator` that implements the ILC once a simulation day.
"""
function DaylyUpdate_reinit(integrator)
	@sync @. integrator.p.hl.daily_background_power += integrator.p.hl.kappa * integrator.p.hl.mismatch_yesterday
	reinit!(integrator, integrator.u; t0=integrator.t, tf=integrator.sol.prob.tspan[2], erase_sol=true)
	nothing
end


function DailyUpdate_X(integrator)
	#println("mismatch ", integrator.p.hl.daily_background_power)
	#println("Q ", integrator.p.hl.Q)
	@sync integrator.p.hl.daily_background_power = integrator.p.hl.Q * (integrator.p.hl.daily_background_power + integrator.p.hl.kappa * integrator.p.hl.mismatch_yesterday)
	#println("mismatch ", integrator.p.hl.daily_background_power)
	nothing
end

end
