using Plots
using Interpolations

N = 1
num_days = 2
Td = 24 * 3600
Tq = 3600 / 4
demand_amp = rand() .* 100.
periodic_demand = t -> demand_amp .* sin(t * pi / (24*3600))^2
inter = interpolate([20. * randn() for i in 1:(num_days*Td/Tq  + 1)], BSpline(Linear()))
residual_demand = t -> inter(1. + t / Tq) # 1. + is needed to avoid trying to access out of range
f = t -> residual_demand(t) .+ periodic_demand(t)

begin
        plot(map(f, 0:1:num_days*Td),
                    xformatter = x->string(x/Td)[1:3],
                    dpi=150,
                    tickfontsize=20,
                    guidefontsize=22,
                    legendfontsize=11,
                    linewidth=2,
                    framestyle=:box,
                    tickdirection=:in,
                    label="total demand"
                    )
        plot!(map(periodic_demand, 0:1:num_days*Td),
                linewidth=3,
                label="baseline"
                )
        xticks!(0:Td÷2:num_days*Td)
        xlabel!("time [d]")
        ylabel!("power demand [W]")
end

savefig("plots/demand.pdf")
