@doc """
This is a module that contains functions with observables for numerical experiments.

# Examples
```julia-repl
julia> include("src/observables.jl")
Main.observables
```
"""
module observables

using DifferentialEquations # import solution types
using StatsBase # for the variance
using Statistics


function frequency_exceedance_ruthlia(sol::DESolution, freq_idx, threshold)
    idx_exc = findall(abs.(sol[freq_idx,:]) .> threshold)
    idx_t = [id[2] for id in idx_exc]
    # idx_exc_N1 = idx_exc[findall([id[1] for id in idx_exc] .== 1)] # only node 1 exceedance indices in CartesianIndex()
    # idx_exc_t_N1 = [id[2] for id in idx_exc_N1] # only node 1 exceedance: index for sol.t
    lbsearch = true
    bounds = [1 1]
    lb = 0
    for i = 1:length(idx_t)-1
        if idx_t[i+1]-idx_t[i] <= 1 && (lbsearch == true)
            lb = idx_t[i]
            lbsearch = false
        elseif idx_t[i+1]-idx_t[i] > 1 && (lbsearch == false)
            bounds = [bounds; [lb idx_t[i]]]
            lbsearch = true
        end
    end
    # single values with exceeded frequency are neglected
    if lbsearch == false
        bounds = [bounds; [lb idx_t[end]]]
    end

    exceedance = sum((sol.t[bounds[:,2]] - sol.t[bounds[:,1]]))
end

function frequency_exceedance_tobi(sol::DESolution, freq_idx, threshold)
    idx_exc = findall(abs.(sol[freq_idx,:]) .> threshold)
    idx_t = [id[2] for id in idx_exc]
    bounds = [1 1]
    lb = 0
    for i = 1:length(idx_t)-1 # includes single values
        if lb == 0
            lb = idx_t[i]
        end
        if idx_t[i+1] - idx_t[i] > 1
            bounds = [bounds; [lb idx_t[i]+1]]
            lb = 0
        end
    end
    if lb != 0
        bounds = [bounds; [lb idx_t[end]]]
    end
    exceedance = sum((sol.t[bounds[:,2]] - sol.t[bounds[:,1]]))
end

function freq_exceedance_frank1(sol, filter, threshold)
    # Using views is important, otherwise every instance of sol[filter,i]
    # allocates a new temporary array.
    @views begin
    ex_t = 0

    if any(x -> abs(x) > threshold, sol[filter, 1])
        ex_t += sol.t[2] - sol.t[1]
    end

    if any(x -> abs(x) > threshold, sol[filter, end])
        ex_t += sol.t[end] - sol.t[end-1]
    end

    for i in 2:length(sol.t)-1
        if any(x -> abs(x) > threshold, sol[filter, i])
            ex_t += sol.t[i + 1] - sol.t[i - 1]
        end
    end

    ex_t/2
    end
end

function freq_exceedance_frank2(sol, filter, threshold)
    # Using views is important, otherwise every instance of sol[filter,i]
    # allocates a new temporary array.
    @views begin
    ex_t = 0
    #This variable records whether we are currently in a region with exceedance
    in_ex_region = false
    step_in_region = false

    if any(x -> abs(x) > threshold, sol[filter, 1])
        # The solution already starts in an exceedance region
        in_ex_region = true
        # we average the neighbouring time steps, /2 is at the end
        ex_t -= sol.t[1] + sol.t[1]
    end

    for i in 2:length(sol.t)
        step_in_region = any(x -> abs(x) > threshold, sol[filter, i])
        if step_in_region && !in_ex_region
            # we are entering an exccedance region
            in_ex_region = true
            # we average the neighbouring time steps, /2 is at the end
            ex_t -= sol.t[i] + sol.t[i-1]
        elseif !step_in_region && in_ex_region
            # we are exciting an exccedance region
            in_ex_region = false
            # we average the neighbouring time steps, /2 is at the end
            ex_t += sol.t[i] + sol.t[i-1]
        end
    end

    if in_ex_region
        # The solution ends in an exceedance region
        # we average the neighbouring time steps, /2 is at the end
        ex_t += sol.t[end] + sol.t[end]
    end

    ex_t / 2
    end
end

@doc """
    frequency_exceedance(time_series, threshold, freq_idx)

    Calculate the frequency exceedance, i.e. the time a frequency `time_series` exceeds a given `threshold`.

    """
function frequency_exceedance(sol, filter, threshold)
    # Using views is important, otherwise every instance of sol[filter,i]
    # allocates a new temporary array.
    @views begin
        ex_t = 0
        #This variable records whether we are currently in a region with exceedance
        in_ex_region = false
        step_in_region = false

        if any(x -> abs(x) > threshold, sol[filter, 1])
            # The solution already starts in an exceedance region
            in_ex_region = true
            # we average the neighbouring time steps, /2 is at the end
            ex_t -= sol.t[1] + sol.t[1]
        end

        for i in 2:length(sol.t)
            step_in_region = any(x -> abs(x) > threshold, sol[filter, i])
            if !in_ex_region && step_in_region
                # we are entering an exccedance region
                in_ex_region = true
                # we average the neighbouring time steps, /2 is at the end
                ex_t -= sol.t[i] + sol.t[i-1]
            elseif in_ex_region && !step_in_region
                # we are exciting an exccedance region
                in_ex_region = false
                # we average the neighbouring time steps, /2 is at the end
                ex_t += sol.t[i] + sol.t[i-1]
            end
        end

        if in_ex_region
            # The solution ends in an exceedance region
            # we average the neighbouring time steps, /2 is at the end
            ex_t += sol.t[end] + sol.t[end]
        end

        ex_t / 2
    end
end


function sum_abs_energy_last_days(sol, energy_filter, n_days)
	t_end = sol.prob.tspan[2]
	total = sol(t_end)[energy_filter]
	total .= 0.
	for t in (t_end - floor(n_days*24*3600)):t_end
		total += abs.(sol(t)[energy_filter])
	end
	total ./= n_days*24*3600
	total
end

function var_last_days(sol, state_filter, n_days)
	t_end = sol.prob.tspan[2]
	tail = [sol(t)[state_filter] for t in (t_end - n_days*24*3600+1):t_end]
	var(tail)
end


@doc """ energy_cost(sol, energy_filter, energy_abs_filter, num_days)
calculates the cost of energy reconstructing ILC power from DailyUpdate_<exp>(), HourlyUpdateEcon() or HourlyUpdateEcon(variant_2=true) from the solution
"""

function energy_cost(sol, energy_filter, energy_abs_filter, num_days, obs_days, lambda)
    num_days = Int(num_days)
    obs_days = Int(obs_days)
    hourly_energy_abs = zeros(24 * num_days, sol.prob.p.N)
    for i = 1:24*num_days
        for j = 1:sol.prob.p.N
            hourly_energy_abs[i, j] = sol(i * 3600.0)[energy_abs_filter[j]]
        end
    end

    sum_lambda = 3600.0 * (lambda - 1)
    sum_sign = zeros(24 * num_days, sol.prob.p.N)
    for i = 1:24*num_days
        for t = (i-1)*3600.0+1.0:i*3600.0
            sum_sign[i, :] .+= sign.(sol(t)[energy_filter])
        end
    end
    sign_term = lambda .* sum_sign .+ sum_lambda

    ILC_power = zeros(num_days + 2, 24, sol.prob.p.N)
    for j = 1:sol.prob.p.N
        if sol.prob.kwargs.data.callback.discrete_callbacks[1].affect!.f.variant_2
            ILC_power[2, :, j] = zeros(24, 1) +
                                 sol.prob.p.hl.kappa * sign_term[1:24, j] .*
                                 hourly_energy_abs[1:24, j] ./ 3600.0
        else
            ILC_power[2, :, j] = zeros(24, 1) +
                                 sol.prob.p.hl.kappa * sign_term[1:24, j]
        end
    end
    for i = 2:num_days
        for j = 1:sol.prob.p.N
            if sol.prob.kwargs.data.callback.discrete_callbacks[1].affect!.f.variant_2
                ILC_power[i+1, :, j] = ILC_power[i, :, j] +
                                       sol.prob.p.hl.kappa *
                                       sign_term[(i-1)*24+1:i*24, j] .*
                                       hourly_energy_abs[(i-1)*24+1:i*24, j] ./
                                       3600.0
            else
                ILC_power[i+1, :, j] = ILC_power[i, :, j] +
                                       sol.prob.p.hl.kappa *
                                       sign_term[(i-1)*24+1:i*24, j]
            end # new update law
        end
    end
    ILC_power_hourly = reshape(ILC_power, (num_days + 2) * 24, sol.prob.p.N)
    cost = (1 - lambda) .* sum(
        abs.(ILC_power_hourly[24*(num_days-obs_days):24*num_days, :]),
        dims = 1,
    ) .+
           lambda .* sum(
        abs.(hourly_energy_abs[24*(num_days-obs_days):24*num_days, :]),
        dims = 1,
    ) ./ 3600.0
end


function energy_cost_II(
    sol,
    energy_filter,
    energy_abs_filter,
    num_days,
    obs_days,
	lambda,
)
    num_days = Int(num_days)
    obs_days = Int(obs_days)
    hourly_energy_abs = zeros(24 * num_days, sol.prob.p.N)
    for i = 1:24*num_days
        for j = 1:sol.prob.p.N
            hourly_energy_abs[i, j] = sol(i * 3600.0)[energy_abs_filter[j]]
        end
    end

    sum_lambda = 3600.0 * (lambda - 1)
    sum_sign = zeros(24 * num_days, sol.prob.p.N)
    for i = 1:24*num_days
        for t = (i-1)*3600.0+1.0:i*3600.0
            sum_sign[i, :] .+= sign.(sol(t)[energy_filter])
        end
    end
    sign_term = lambda .* sum_sign .+ sum_lambda

    ILC_power = zeros(num_days + 2, 24, sol.prob.p.N)

    if sol.prob.kwargs.data.callback.discrete_callbacks[1].affect!.f.variant_2
        cover_values = [(sign_term[1:24, x] .* hourly_energy_abs[1:24, x] ./
                         3600.0 + sum(
            sign_term[:, sol.prob.p.hl.ilc_covers[x]] .*
            hourly_energy_abs[1:24, sol.prob.p.hl.ilc_covers[x]] ./ 3600.0,
            dims = 2,
        )) ./ (1 +
                         length(collect(values(sol.prob.p.hl.ilc_covers[x])))) for x in sol.prob.p.hl.ilc_nodes]
        ILC_power[2, :, sol.prob.p.hl.ilc_nodes] = zeros(
            24,
            length(sol.prob.p.hl.ilc_nodes),
        ) + sol.prob.p.hl.kappa .*
                                                   hcat(Vector(collect(cover_values))...)
    else
        cover_values = [(sign_term[1:24, x] + sum(
            sign_term[1:24, sol.prob.p.hl.ilc_covers[x]],
            dims = 2,
        )) ./ (1 +
                         length(collect(values(sol.prob.p.hl.ilc_covers[x])))) for x in sol.prob.p.hl.ilc_nodes]
        ILC_power[2, :, sol.prob.p.hl.ilc_nodes] = zeros(
            24,
            length(sol.prob.p.hl.ilc_nodes),
        ) + sol.prob.p.hl.kappa .*
                                                   hcat(Vector(collect(cover_values))...)
    end

    for i = 2:num_days
        if sol.prob.kwargs.data.callback.discrete_callbacks[1].affect!.f.variant_2
            cover_values = [(sign_term[(i-1)*24+1:i*24, x] .*
                             hourly_energy_abs[(i-1)*24+1:i*24, x] ./ 3600.0 +
                             sum(
                sign_term[(i-1)*24+1:i*24, sol.prob.p.hl.ilc_covers[x]] .*
                hourly_energy_abs[
                    (i-1)*24+1:i*24,
                    sol.prob.p.hl.ilc_covers[x],
                ] ./ 3600.0,
                dims = 2,
            )) ./ (1 +
                             length(collect(values(sol.prob.p.hl.ilc_covers[x])))) for x in sol.prob.p.hl.ilc_nodes]
            ILC_power[i+1, :, sol.prob.p.hl.ilc_nodes] = ILC_power[
                i,
                :,
                sol.prob.p.hl.ilc_nodes,
            ] + sol.prob.p.hl.kappa .*
                                                         hcat(Vector(collect(cover_values))...)
        else
            cover_values = [(sign_term[(i-1)*24+1:i*24, x] + sum(
                sign_term[(i-1)*24+1:i*24, sol.prob.p.hl.ilc_covers[x]],
                dims = 2,
            )) ./ (1 +
                             length(collect(values(sol.prob.p.hl.ilc_covers[x])))) for x in sol.prob.p.hl.ilc_nodes]
            ILC_power[i+1, :, sol.prob.p.hl.ilc_nodes] = ILC_power[
                i,
                :,
                sol.prob.p.hl.ilc_nodes,
            ] + sol.prob.p.hl.kappa .*
                                                         hcat(Vector(collect(cover_values))...)
        end
    end
    ILC_power_hourly = reshape(ILC_power, (num_days + 2) * 24, sol.prob.p.N)
    cost = (1 - lambda) .* sum(
        abs.(ILC_power_hourly[24*(num_days-obs_days):24*num_days, :]),
        dims = 1,
    ) .+
           lambda .* sum(
        abs.(hourly_energy_abs[24*(num_days-obs_days):24*num_days, :]),
        dims = 1,
    ) ./ 3600.0
end


function energy_cost_III(
    sol,
    energy_filter,
    energy_abs_filter,
    num_days,
    obs_days,
	lambda,
)
    num_days = Int(num_days)
    obs_days = Int(obs_days)
    hourly_energy_abs = zeros(24 * num_days, sol.prob.p.N)
    for i = 1:24*num_days
        for j = 1:sol.prob.p.N
            hourly_energy_abs[i, j] = sol(i * 3600.0)[energy_abs_filter[j]]
        end
    end

    sum_lambda = 3600.0 * (lambda - 1)
    sum_sign = zeros(24 * num_days, sol.prob.p.N)
    for i = 1:24*num_days
        for t = (i-1)*3600.0+1.0:i*3600.0
            sum_sign[i, :] .+= sign.(sol(t)[energy_filter])
        end
    end
    sign_term = lambda .* sum_sign .+ sum_lambda

    ILC_power = zeros(num_days + 2, 24, sol.prob.p.N)

    if sol.prob.kwargs.data.callback.discrete_callbacks[1].affect!.f.variant_2
        ILC_power[2, :, sol.prob.p.hl.ilc_nodes] = zeros(
            24,
            length(sol.prob.p.hl.ilc_nodes),
        ) +
                                                   sol.prob.p.hl.kappa *
                                                   sign_term[
            1:24,
            sol.prob.p.hl.ilc_nodes,
        ] .* hourly_energy_abs[
            1:24,
            sol.prob.p.hl.ilc_nodes,
        ] ./ 3600.0
    else
        ILC_power[2, :, sol.prob.p.hl.ilc_nodes] = zeros(
            24,
            length(sol.prob.p.hl.ilc_nodes),
        ) + sol.prob.p.hl.kappa *
                                                   sign_term[
            1:24,
            sol.prob.p.hl.ilc_nodes,
        ]
    end

    for i = 2:num_days
        if sol.prob.kwargs.data.callback.discrete_callbacks[1].affect!.f.variant_2
            ILC_power[i+1, :, sol.prob.p.hl.ilc_nodes] = ILC_power[
                i,
                :,
                sol.prob.p.hl.ilc_nodes,
            ] +
                                                         sol.prob.p.hl.kappa *
                                                         sign_term[
                (i-1)*24+1:i*24,
                sol.prob.p.hl.ilc_nodes,
            ] .* hourly_energy_abs[
                (i-1)*24+1:i*24,
                sol.prob.p.hl.ilc_nodes,
            ] ./ 3600.0
        else
            ILC_power[i+1, :, sol.prob.p.hl.ilc_nodes] = ILC_power[
                i,
                :,
                sol.prob.p.hl.ilc_nodes,
            ] +
                                                         sol.prob.p.hl.kappa *
                                                         sign_term[
                (i-1)*24+1:i*24,
                sol.prob.p.hl.ilc_nodes,
            ]
        end # new update law
    end
    ILC_power_hourly = reshape(ILC_power, (num_days + 2) * 24, sol.prob.p.N)
    cost = (1 - lambda) .* sum(
        abs.(ILC_power_hourly[24*(num_days-obs_days):24*num_days, :]),
        dims = 1,
    ) .+
           lambda .* sum(
        abs.(hourly_energy_abs[24*(num_days-obs_days):24*num_days, :]),
        dims = 1,
    ) ./ 3600.0
end


function energy_cost_IV(
    sol,
    energy_filter,
    energy_abs_filter,
    num_days,
    obs_days,
	lambda,
)
    num_days = Int(num_days)
    obs_days = Int(obs_days)
    hourly_energy_abs = zeros(24 * num_days, sol.prob.p.N)
    for i = 1:24*num_days
        for j = 1:sol.prob.p.N
            hourly_energy_abs[i, j] = sol(i * 3600.0)[energy_abs_filter[j]]
        end
    end

    sum_lambda = 3600.0 * (lambda - 1)
    sum_sign = zeros(24 * num_days, sol.prob.p.N)
    for i = 1:24*num_days
        for t = (i-1)*3600.0+1.0:i*3600.0
            sum_sign[i, :] .+= sign.(sol(t)[energy_filter])
        end
    end
    sign_term = lambda .* sum_sign .+ sum_lambda

    ILC_power = zeros(num_days + 2, 24, sol.prob.p.N)
    if sol.prob.kwargs.data.callback.discrete_callbacks[1].affect!.f.variant_2
        ILC_power[2, :, :] = zeros(24, sol.prob.p.N) +
                             sign_term[1:24, :] .* hourly_energy_abs[1:24, :] ./
                             3600.0 * sol.prob.p.hl.kappa'
    else
        ILC_power[2, :, :] = zeros(24, sol.prob.p.N) +
                             sign_term[1:24, :] * sol.prob.p.hl.kappa'
    end

    for i = 2:num_days
        if sol.prob.kwargs.data.callback.discrete_callbacks[1].affect!.f.variant_2
            ILC_power[i+1, :, :] = ILC_power[i, :, :] +
                                   sign_term[(i-1)*24+1:i*24, :] .*
                                   hourly_energy_abs[(i-1)*24+1:i*24, j] ./
                                   3600.0 * sol.prob.p.hl.kappa'
        else
            ILC_power[i+1, :, :] = ILC_power[i, :, :] +
                                   sign_term[(i-1)*24+1:i*24, :] *
                                   sol.prob.p.hl.kappa'
        end # new update law
    end
    ILC_power_hourly = reshape(ILC_power, (num_days + 2) * 24, sol.prob.p.N)
    cost = (1 - lambda) .* sum(
        abs.(ILC_power_hourly[24*(num_days-obs_days):24*num_days, :]),
        dims = 1,
    ) .+
           lambda .* sum(
        abs.(hourly_energy_abs[24*(num_days-obs_days):24*num_days, :]),
        dims = 1,
    ) ./ 3600.0
end

end
