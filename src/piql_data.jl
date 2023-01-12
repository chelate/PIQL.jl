using StatsBase: weights, Weights, sample # for Weights
using StatsFuns: logsumexp
export run_piql!, backpropagate_weights!, initial_piql, EnergyEstimate, training_epoch!

struct EnergyEstimate{S,A}
    state::S
    action::A
    β::Float64
    xi::Float64 # energy fluctuation realization
    logz::Float64 # previous log z
end 

"""
for reference: current stateaction definition
struct StateAction{S,A} # static and constructed on forward pass
    state::S
    action::A
    β::Float64 # the beta under which the state was generated
    E_actor::Float64 # the actor energy associated with the current state
    E_critic::Float64 # the critic energy associated with the previous state
    
    # 
    # diagnostics
    #

    cost::Float64 # actually incurred cost
    f::Float64 # free energy of current action
    u::Float64 # average energy of current action
end

currently assumption that energy is constant over run.
"""
mutable struct PiqlParticle{S,A}
    worldline::Vector{StateAction{S,A}} # currently evolving state buffer
    memory::Vector{EnergyEstimate{S,A}} # list of actively evolving nodes
    time::Int # time since worline began (not time in state) for iteration purposes
    depth::Int # the target depth of PIQL back propagation for assessing bias/variance tradeoff
    function PiqlParticle(stateaction::StateAction{S,A}; depth = 1) where {S,A} 
        worldline = [stateaction]
        memory = EnergyEstimate{S,A}[]
        return new{S,A}(worldline, memory, 1, depth)
    end
end

function initial_piql(ctrl::ControlProblem, actor; depth = 1)
    sa = intial_state_action(ctrl, actor)
    # doesn't yet have a valid critic energy
    return PiqlParticle(sa; depth)
end

function run_piql!(piql, ctrl, actor)
    sa = piql.worldline[piql.time]
    terminate_early = (piql.time > piql.depth)
    # for depth = 1, you must still go at least one step,
    if ctrl.terminal_condition(sa.state)
        backpropagate_weights!(piql, ctrl)
        piql.worldline[1] = intial_state_action(ctrl, actor)
        terminated = true
    elseif terminate_early
        backpropagate_weights!(piql, ctrl)
        piql.worldline[1] = sa # restart piql from last position
        terminated = true
    else 
        piql.time += 1
        if length(piql.worldline) < piql.time
            resize!(piql.worldline, piql.time)
        end # we assume not terminal
        piql.worldline[piql.time] = new_state_action(sa, ctrl, actor)
        terminated = false
    end
    return terminated
end

function energy_estimate(sa0, sa1, logz, ctrl)
    xi = sa1.E_critic - log1p(ctrl.γ * expm1(logz)) / sa1.β # not quite sure which β to use
    new_logz = sa0.β*(sa0.E_actor - xi) # finish the recurrence relation
    return (EnergyEstimate(sa0.state, sa0.action, sa1.β, xi, logz), new_logz)
end

function backpropagate_weights!(piql, ctrl)
    # may use ctrl later to update things with changing $β$.
    logz = 0.0 # starting z
    while piql.time > 1
        sa1 = piql.worldline[piql.time]
        piql.time -= 1
        sa0 = piql.worldline[piql.time] # previous state
        (ee, logz) = energy_estimate(sa0, sa1, logz, ctrl) # use the last logz
        # ee is a function of sa0 state acton pair
        push!(piql.memory, ee)
    end
    if piql.time != 1
        error("didn't make it to the end")
    end
end

function random_piql(ctrl, actor; depth = 1)
    piql = initial_piql(ctrl::ControlProblem, actor; depth)
    while true 
        terminated = run_piql!(piql, ctrl, actor)
        if terminated 
            break
        end
    end
    return piql
end

function training_epoch!(piql, ctrl, actor)
    # may use ctrl later to update things with changing $β$.
    while true 
        terminated = run_piql!(piql, ctrl, actor)
        if terminated 
            break
        end
    end
    train!(actor, piql.memory)
end
