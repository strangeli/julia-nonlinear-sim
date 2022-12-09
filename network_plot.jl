using LightGraphs
using GraphPlot
using Colors
dir = @__DIR__

function neighbour_map(g, vc)
    Dict([v => neighbors(g, v) for v in vc])
end

function ilc_update(control_energy, g)
    # set of vertices that are independent
    # (no two vertices are adjacent to each other)
    ilc_nodes = independent_set(g, DegreeIndependentSet())
    nm = neighbour_map(g, ilc_nodes)
    return Dict([x => control_energy[x] + sum(control_energy[nm[x]]) for x in ilc_nodes])
end



n = 24
g = random_regular_graph(iseven(3n) ? n : (n-1), 3)

# all control energy observations
control_energy = rand(n)

# low level output from ILC-controlled nodes and
# the sum of their neighbours' values
y_l = ilc_update(control_energy, g)
# Dict für index and value; keys(y_l) is indices, values(y_l) = control energy is the energy at node + its neighbors
# im Callback: graph und function: define y_l - graph aus integrator.p.incidence oder graph in p integrieren, vertex function into network dynamics
# P[keys(y_l)] = P_yesterday[keys] + kappa * values(y_l)

# access the values via
values(y_l)

# nodes membership
membership = [v ∈ keys(y_l) ? 1 : 2 for v in vertices(g)];
nodecolor = [colorant"lightseagreen", colorant"orange"];
nodefillc = nodecolor[membership];
nodelabel = [v ∈ keys(y_l) ? "C" : "" for v in vertices(g)];

gplot(g, nodelabel=nodelabel, nodefillc=nodefillc)
