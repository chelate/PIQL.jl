using Revise
using PIQL
using StatsBase


gw = make_gridworld([20,20]; density = 0.15);
ctrl = make_ctrl(gw);
states = state_iterator(gw)
actor0 = get_ideal_actor(ctrl, states)
actor1 = jitter_actor(actor0, 1.0)

function energy_est_truth(actor0, actor1; depth = 1)
    piql = PIQL.random_piql(ctrl, actor1; depth)
    ee = last(piql.memory)
    (actor0(ee.state, ee.action), ee.xi)
end

energy_est_truth(actor0, actor1; depth = 1)

"""
For plotting the result, show the free energy as a function of state
This might be helpful.
"""
function free_energy_matrix(ctrl, actor, gw; max = 60.0)
    out = fill(NaN,size(gw.walls))
    states = state_iterator(gw)
    for ii in states
        state = collect(Tuple(ii))
        out[ii] = PIQL.free_energy(state,ctrl,actor)
    end
    out[out .> max] .= max
    out[isnan.(out)] .= max
    return out
end

free = free_energy_matrix(ctrl, actor0, gw)
free1 = free_energy_matrix(ctrl, actor1, gw)

run_piql!(piql, ctrl, actor)

training_epoch!(piql, ctrl, actor)


for ii in 1:100
    training_epoch!(piql, ctrl, actor)
end

using UnicodePlots
a = zeros(size(gw.walls))
a[gw.goal...] += 2
a .+= gw.walls
heatmap(a)
heatmap(free)
heatmap(free1)