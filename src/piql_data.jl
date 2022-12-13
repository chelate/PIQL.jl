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

function energy_estimate(sa, logz)
    xi = sa.E_critic - logz / sa.β
    ee = EnergyEstimate(sa.state, sa.action, sa.β, xi, logz)
end


mutable struct PiqlParticle{S,A}
    worldline::Vector{StateAction{S,A}} # currently evolving state buffer
    memory::Vector{EnergyEstimate{S,A}} # list of actively evolving nodes
    time::Int # time since worline began (not time in state) for iteration purposes
    function PiqlParticle(stateaction::StateAction{S,A}) where {S,A} 
        worldline = [stateaction]
        memory = EnergyEstimate{S,A}[]
        return new{S,A}(worldline, memory, 1)
    end
end

function initial_piql(ctrl::ControlProblem, actor)
    sa = intial_state_action(ctrl, actor)
    return PiqlParticle(sa)
end

function run_piql!(piql, ctrl, actor)
    sa = piql.worldline[piql.time]
    terminate_early = rand() > ctrl.γ
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

function backpropagate_weights!(piql, ctrl)
    # may use ctrl later to update things with changing $β$.
    logz = 0.0 # starting z
    while piql.time > 1
        sa = piql.worldline[piql.time]
        ee = energy_estimate(sa, logz)
        push!(piql.memory, ee)
        piql.time -= 1
        sa = piql.worldline[piql.time]
        logz = sa.β*(sa.E_actor - ee.xi)
    end
    if piql.time != 1
        error("didn't make it to the end")
    end
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
