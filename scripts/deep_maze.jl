
begin
using PIQL
using StatsBase
end
using SimpleChains

begin 
gw = make_gridworld([20,20]; density = 0.30);
ctrl = make_ctrl(gw; γ = .99);
states = state_iterator(gw);
actor0 = get_ideal_actor(ctrl, states; β = 0.2);
actor_jittered = jitter_actor(actor0, ctrl, states; jitter =  0.5);
actor_heated = get_ideal_actor(ctrl, states; β = 2.0);
actor_heated.β = 1.0;
actor_cooled = get_ideal_actor(ctrl, states; β = 0.5);
actor_cooled.β = 1.0;
vectorstates = collect.(Tuple.(states))
end

chain_actor = init_chainpv(ctrl, 
    TurboDense(tanh, 32); sdims = 2, β = 0.2)

chain_snapshot = tabularize(ctrl,chain_actor,vectorstates)
excess_reward(ctrl, actor0, tabularize(ctrl,chain_actor,vectorstates), states)

out = training_curve(chain_actor, ctrl, actor0, states)

function training_curve(chain_actor, ctrl, actor0, states; epochs = 4*10^3, depth = 50)
    vectorstates = collect.(Tuple.(states))
    piql = PIQL.random_piql(ctrl, chain_actor; depth)
    out = Float64[]
    for ii in 1:epochs
        training_epoch!(piql, ctrl, chain_actor)
        if  1 == mod(ii,100) 
            c = excess_reward(ctrl, actor0, tabularize(ctrl,chain_actor,vectorstates), states)
            push!(out,c)
        end
    end
    return out
end

function free_energy_matrix(ctrl, actor, gw; min = -44.0)
    out = fill(NaN,size(gw.walls))
    states = state_iterator(gw)
    for ii in states
        state = collect(Tuple(ii))
        out[ii] = PIQL.value_function(state,ctrl,actor)
    end
    out[out .< min] .= min
    out[isnan.(out)] .= min .* 1.1
    return out
end

using UnicodePlots
free = free_energy_matrix(ctrl, chain_snapshot, gw; min = -101.0)
heatmap(free)

free0 = free_energy_matrix(ctrl, actor0, gw)
heatmap(free0)
