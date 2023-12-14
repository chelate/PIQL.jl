module PIQL
export ControlProblem, average_reward



"""
ControlProblem is a struct with fields that are 
    the functions which completely define a KL-control problem
"""
struct ControlProblem{AA, U, P, R, PA, T, W}
    action_space::AA # something that we can iterate over
    action_prior::U # π(s,a) -> Float64 exactly like energy, assumed to be normalized
    propagator::P # p(x0, a) -> x1 ("random" state)
    reward_function::R # r(x0, a, x1) -> reward ::Float64
    # given in entropic units already
    propagator_average::PA # (s,a,f) -> K·f
    terminal_condition::T # T(x) -> bdol
    initial_state::W # W() -> x0 generates inital states of interest
    γ::Float64 # positive number less than one discount over time
end
# Write your package code here.


"""
the averaged reward
"""
function average_reward(ctrl,s,a)
    ctrl.propagator_average(s,a, s1 -> ctrl.reward_function(s,a,s1))
end


"""
generate `controlv` function. 
"""
function controlV(ctrl, V, s, a)
    ctrl.γ*ctrl.propagator_average(s,a,V) - value_function(s) + 
        average_reward(ctrl,s,a)
end

"""
the normalization for an unnormalized q function, function of state
"""
function Qnormalization(ctrl, Q, s)
    z = 0.0
    for a in action_space
        z += ctrl.prior(s,a)*exp(Q(s, a)) 
    end
    return log(z)
end

"""
"""
function controlQ(ctrl, Q, s, a)
    Q(s,a) - Qnormalization(ctrl, Q, s)
end

function action_probability(ctrl, Q, s, a)
    ctrl.prior(s,a) * exp(controlQ(ctrl,Q,s,a))
end

function fitness(ctrl,V, Q, s, a)
    controlV(ctrl, V, s, a) - controlQ(ctrl, Q, s, a)
end

"""
this is used to generate the optimal control directly, it is the backup operator fo rhte bellman equation
see eq. 7 in  paper 
"""
function truevalue_recursion(ctrl, s, ν)
    z = 0.0
    for a in action_space
        z += ctrl.prior(s,a)*exp(ctrl.γ * ctrl.propagator_average(s, a, ν) + average_reward(ctrl, s, a)) 
    end
    return log(z)
end

"""
this is used to detemrine Z
"""
function z_recursion(ctrl, Q, V, s, Z; β= 1.0)
    out = 0.0
    for a in action_space
        out += action_probability(ctrl, Q, s, a) * exp(β * fitness(ctrl, V, Q, s, a) ) * propagator_average(ctrl, s, a, Z)
    end
    return out
end    

include("value_iteration.jl")
include("Problems/BinaryBandit.jl")
include("Problems/GridWorld.jl")

end
