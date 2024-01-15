


"""
the statistic is a function with sig. 
    f(s,a,ss) 
"""

#using ResumableFunctions
using StatsBase

function choose_action(ctrl, s, Q)
    w = Weights([action_probability(ctrl, Q, s, a) for a in action_space])
    a = sample(ctrl.action_space, w)
end

klterm(x,y) = xlogx(x) - xlogy(x,y)
function kl_cost(ctrl, s, Q) 
    sum( klterm(action_probability(ctrl, Q, s, a), ctrl.action_prior(s,a)) for a in ctrl.action_space)
end


@resumable function path_transitions(ctrl, s0, Q, ; nsteps = 20)
    s = s0
    for _ in 1:nsteps
        a = choose_action(ctrl, s, Q)
        ss = ctrl.propagator(s,a)
        @yield (s,a,ss)
        s = ss 
    end
    return out
end

function fitness_reward_estimate(ctrl, s0, V, Q)
    total = V(s0)
    out = Float64[]
    for (s,a,ss) in path_transitions(ctrl, s0, Q, ; nsteps = 20)
        total += fitness(ctrl, V, Q, s, a)
        push!(out, total)
    end
end

function direct_reward_estimate(ctrl, s0, V, Q)
    total = V(s0)
    out = Float64[]
    for (s,a,ss) in path_transitions(ctrl, s0, Q, ; nsteps = 20)
        total += ctrl.reward(s, a, ss) - kl_cost(ctrl, s, Q) 
        push!(out, total)
    end
end


