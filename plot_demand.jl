using Plots
#inspectdr()
plotlyjs()
using Interpolations
using CSV
dir = @__DIR__

N = 1
num_days = 1
Td = 24 * 3600
Tq = 3600 / 4
demand_amp = 80.
periodic_demand = t -> demand_amp .* sin(t * pi / (24*3600))^2
inter = interpolate([20. * randn() for i in 1:(num_days*Td/Tq  + 1)], BSpline(Linear()))
residual_demand = t -> inter(1. + t / Tq) # 1. + is needed to avoid trying to access out of range
f = t -> residual_demand(t) .+ periodic_demand(t)

begin
        dem = map(f, 0:1:num_days*Td)
        plot(100 .* dem ./ maximum(dem),
                    xformatter = x->string(Int(24 * x/Td)),
                    dpi=150,
                    tickfontsize=12,
                    guidefontsize=15,
                    legendfontsize=12,
                    linewidth=2,
                    framestyle=:box,
                    tickdirection=:in,
                    legend=:topleft,
                    label="residual demand"
                    )
        per = map(periodic_demand, 0:1:num_days*Td)
        plot!(100 .* per ./ maximum(dem),
                linewidth=3,
                linestyle = :dash,
                label="baseline"
                )
        #plot!(1:Td, 100 .* average_weekday' ./ maximum(average_weekday),
        #        label="empirical")
        xticks!(0:Td÷12:num_days*Td)
        xlabel!("time [h]")
        ylabel!("power demand [% peak]")
end

savefig("$dir/plots/demand.tex")

#
# # real data
# # resolution 1Hz, duration 1 week, unit 1W
# load = CSV.read("input_data/Load_Power.txt", header=[:P], datarow=1, types=Dict(1=>Float64));
# l_minute = 60
# l_hour = l_minute * 60
# l_day = l_hour * 24
# n_days = length(load[:P]) ÷ l_day
#
# M = zeros(n_days, l_day)
# for d in 1:n_days
#     day_range = 1 + (d-1)*l_day : d * l_day
#     M[d, :] .= load.P[day_range]
# end
#
# average_weekday = sum(M[1:5,:], dims=1) ./ 5
