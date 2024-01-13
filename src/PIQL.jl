module PIQL
export ControlProblem, average_reward, modify
export generate_ν, generate_z, generate_logz, z_updateV, logz_updateV, generate_optimality_gap# from value_iteration.jl
import LogExpFunctions: logsumexp, logaddexp


"""
ControlProblem is a struct with fields that are 
    the functions which completely define a KL-control problem
"""
struct ControlProblem{AA, U, P, R, PA, PE, T, W}
    action_space::AA # something that we can iterate over
    action_prior::U # π(s,a) -> Float64 exactly like energy, assumed to be normalized
    propagator::P # p(x0, a) -> x1 ("random" state)
    reward_function::R # r(x0, a, x1) -> reward ::Float64
    # given in entropic units already
    propagator_average::PA # (s,a,f) -> K·f
    logpropagatorexp::PE
    terminal_condition::T # T(x) -> bdol
    initial_state::W # W() -> x0 generates inital states of interest
    γ::Float64 # positive number less than one discount over time
end
# Write your package code here.

function modify(ctrl;
    action_space = ctrl.action_space,
    action_prior = ctrl.action_prior,
    propagator = ctrl.propagator,
    reward_function = ctrl.reward_function,
    propagator_average = ctrl.propagator_average,
    logpropagatorexp = ctrl.logpropagatorexp,
    terminal_condition = ctrl.terminal_condition,
    initial_state = ctrl.initial_state,
    γ = ctrl.γ)
    ControlProblem(
    action_space,
    action_prior,
    propagator,
    reward_function,
    # given in entropic units already
    propagator_average,
    logpropagatorexp,
    terminal_condition,
    initial_state,
    γ) # positive number less than one discount over time
end


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
    ctrl.γ*ctrl.propagator_average(s,a,V) - V(s) + 
        average_reward(ctrl,s,a)
end

"""
the normalization for an unnormalized q function, function of state
    needs to be stabilized
"""
function Qnormalization(ctrl, Q, s)
    logsumexp(log(ctrl.action_prior(s,a)) + Q(s, a) 
        for a in ctrl.action_space)
end

"""
"""
function controlQ(ctrl, Q, s, a)
    Q(s,a) - Qnormalization(ctrl, Q, s)
end

function action_probability(ctrl, Q, s, a)
    ctrl.action_prior(s,a) * exp(controlQ(ctrl,Q,s,a))
end

function fitness(ctrl,V, Q, s, a)
    controlV(ctrl, V, s, a) - controlQ(ctrl, Q, s, a)
end

"""
this is used to generate the optimal control directly, it is the backup operator fo rhte bellman equation
see eq. 7 in  paper 
"""
function truevalue_recursion(ctrl, s, ν)
    logsumexp(
        log(ctrl.action_prior(s,a)) + 
        ctrl.γ * ctrl.propagator_average(s, a, ν) +
        average_reward(ctrl, s, a)
        for a in ctrl.action_space)
end

"""
this is used to detemrine Z
"""
function z_recursion(ctrl, Q, V, s, Z; β= 1.0)
    out = 0.0
    for a in ctrl.action_space
        out += action_probability(ctrl, Q, s, a) * exp(β * fitness(ctrl, V, Q, s, a) ) * (ctrl.γ * ctrl.propagator_average(s, a, Z) + 1 - ctrl.γ)
    end
    return out
end    

"""
probably not numerically stable enough
"""


function z_recursion2(ctrl, Q, V, s, Z; β= 1.0)
    out = 0.0
    for a in ctrl.action_space
        out += ctrl.action_prior(s,a) * exp(β * controlV(ctrl, V, s, a) + (1-β) * controlQ(ctrl,Q,s,a)) * (ctrl.γ * ctrl.propagator_average(s, a, Z) + 1 - ctrl.γ)
    end
    return out
end


function z_recursion3(ctrl, Q, V, s, Z; β= 1.0)
    out = logsumexp(
    log(ctrl.action_prior(s,a)) + β * controlV(ctrl, V, s, a) + (1-β) * controlQ(ctrl,Q,s,a) + log(ctrl.γ * ctrl.propagator_average(s, a, Z) + 1 - ctrl.γ)
        for a in ctrl.action_space)
    return exp(out)
end   


"""
=   log(ctrl.γ * ctrl.propagator_average(s, a, Z) + 1 - ctrl.γ)
=   log( 1 + ctrl.γ/(1-ctrl.γ) + log(ctrl.propagator_average(s, a, Z))) + log(1 - ctrl.γ)
=   logaddexp( log(ctrl.γ) + ctrl.logpropagator_exp(s, a, logz)) , log(1 - ctrl.γ))
"""

function logz_recursion(ctrl, Q, V, s, logz; β= 1.0)
    out = logsumexp(
    log(ctrl.action_prior(s,a)) + β * controlV(ctrl, V, s, a) + (1-β) * controlQ(ctrl,Q,s,a) + logaddexp( log(ctrl.γ) + ctrl.logpropagatorexp(s, a, logz) , log(1 - ctrl.γ))
        for a in ctrl.action_space)
    return out
end   


include("value_iteration.jl")
include("Problems/BinaryBandit.jl")
include("Problems/GridWorld.jl")

end #MODULE
