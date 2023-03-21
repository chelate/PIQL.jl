using StatsBase: weights, Weights, sample # for Weights
using StatsFuns: logsumexp
export run_piql!, backpropagate_weights!, initial_piql, EnergyEstimate, training_epoch!

struct QEstimate{S,A}
    sa0::StateAction{S,A} # the state action information at this time step
    sa1::StateAction{S,A} # the state action information at this time step
    logz0::Float64 # log z at state in sa0.state
    logz1::Float64 # log z at state in sa1.state
end 


function qbound(qe::QEstimate)
    # equivalent to log(ξ) / β from the paper
    # this is the best estimate for the q function at qe.sa0 
    # given the rest of the trajectory
    qe.sa1.criticq + qe.logz1 / qe.sa1.β
end

"""
current StateAction definition:

struct StateAction{S,A} # static and constructed on forward pass
    # atomic unit of data for all reinforcement learning
    # includes all information and diagnostics available from a single step.
    state::S
    action::A
    β::Float64 # the beta under which the temperature is allowed to fluxuate
    actorq::Float64
    criticq::Float64
    reward::Float64 # actually incurred cost entering the state
    V::Float64 # free energy of current action (observed value)
    U::Float64 # average energy of current action
end
"""
mutable struct PiqlParticle{S,A}
    worldline::Vector{StateAction{S,A}} # currently evolving state buffer
    memory::Vector{QEstimate{S,A}} 
    time::Int # time since worline began (not time in state) for iteration purposes
    depth::Int # the target depth of PIQL back propagation for assessing bias/variance tradeoff
    function PiqlParticle(stateaction::StateAction{S,A}; depth = 1) where {S,A} 
        worldline = [stateaction]
        memory = QEstimate{S,A}[]
        return new{S,A}(worldline, memory, 1, depth)
    end
end

function initial_piql(ctrl::ControlProblem, actor; 
    depth = 1, 
    sa = intial_state_action(ctrl, actor))
    # doesn't yet have a valid critic energy
    return PiqlParticle(sa; depth)
end

"""
Expected dynamics: grow the worldline until the terminal condition.
then backpropogate weights and return terminated = true
"""
function run_piql!(piql, ctrl, actor)
    sa = piql.worldline[piql.time]
    terminate_early = (piql.time > piql.depth)
    # for depth = 1, you must still go at least one step,
    if ctrl.terminal_condition(sa.state)
        backpropagate_weights!(piql, ctrl)
        piql.worldline[1] = initial_state_action(ctrl, actor)
        # if we restart piql, we need to decide how that will work
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


"""
z(t) = exp(β(criticq - actorq)) (γ z(t+1) + (1-γ))
z(t) = exp(β(criticq - actorq)) (γ (z(t+1)-1) + 1)
logz(t) = β(criticq - actorq) +  log(γ*(z(t+1)-1) + 1)
= β(criticq - actorq) +  logp1(γ*expm1(logz(t+1)))
"""
function qestimate(sa0, sa1, logz1, ctrl)
    logz0 = sa0.β * (sa1.criticq - sa0.actorq) + log1p(ctrl.γ * expm1(logz1)) # not quite sure which β to use
    return (QEstimate(sa0, sa1, logz0, logz1), logz0)
end

function backpropagate_weights!(piql, ctrl)
    # may use ctrl later to update things with changing $β$.
    logz = 0.0 # starting z
    while piql.time > 1
        sa1 = piql.worldline[piql.time]
        piql.time -= 1
        sa0 = piql.worldline[piql.time] # previous state
        (qe, logz) = qestimate(sa0, sa1, logz, ctrl) # use the last logz
        # qe is a function of sa0 state acton pair
        push!(piql.memory, qe) # add to the end so that things are backward facing
    end
end

function random_piql(ctrl, actor; depth = 1, 
    sa = initial_state_action(ctrl, actor))
    piql = initial_piql(ctrl::ControlProblem, actor; depth, sa)
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
