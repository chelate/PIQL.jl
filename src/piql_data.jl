using StatsBase: weights, Weights, sample # for Weights
using StatsFuns: logsumexp
export brine_piql, initial_piql, forward_evolve!, rerun_piql

struct EnergyEstimate{S,A}
    state::S
    action::A
    β::Float64
    xi::Float64 # energy fluctuation realization
    logz::Float64
end

function energy_estimate(sa, logz)
    xi = sa.E_critic - logz / sa.β
    ee = EnergyEstimate(sa.state, sa.action, sa.β, xi, logz)
end


mutable struct PiqlParticle{S,A}
    worldline::Vector{StateAction{S,A}} # currently evolving state buffer
    memory::Vector{EnergyEstimate{S,A}} # list of actively evolving nodes
    time::Int
    function PiqlParticle(stateaction::StateAction{S,A}) where {S,A} 
        worldline = [stateaction]
        memory = EnergyEstimate{S,A}[]
        return new(worldline, memory, 1)
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
    elseif terminate_early
        backpropagate_weights!(piql, ctrl)
        piql.worldline[1] = sa # restart piql from last position
    end
    piql.time += 1
    if length(piql.worldline) < piql.time
        resize!(piql.worldline, piql.time)
    end # we assume not terminal
    piql.worldline[piql.time] = new_state_action(sa, ctrl, actor)
end

function backpropagate_weights!(piql, ctrl)
    # may use ctrl later to update things with changing $β$.
    logz = 0 # starting z
    while piql.time > 1
        sa = piql.worldline[time]
        ee = energy_estimat(sa, logz)
        push!(piql.memory, ee)
        piql.time -= 1
        sa = piql.worldline[time]
        logz = sa.β*(sa.E_actor - ee.xi)
    end
    if piql.time == 1
        error("didn't make it to the end")
    end
end
