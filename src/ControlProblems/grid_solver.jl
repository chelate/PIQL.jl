export get_ideal_actor, jitter_actor

"""
J_i = sum_a π_a e^-E_a^i
"""

function update_actor_everywhere!(actor, ctrl, states)
    discr = 0.0
    for ii in states
        state = collect(Tuple(ii))
        for a in ctrl.action_space
            old_energy = actor(state,a)
            new_energy = energy_critic(state, a, ctrl, actor; critic_samples = 1)
            discr += abs(new_energy-old_energy)
            key = actor.mapping(state,a)
            if haskey(actor.energy, key)
                actor.energy[key] = new_energy
            else
                push!(actor.energy, key => new_energy)
            end
        end
    end
    return discr
end

function get_ideal_actor(ctrl, states; β = 1.0, maxiter = 1e5)
    actor = init_tabular_actor_piql(ctrl; β)
    iter  = 0
    while true
        iter +=1
        discr = update_actor_everywhere!(actor, ctrl, states)
        if (discr < 1e-9) | (iter > maxiter) 
            println.([discr,iter])
            break
        end
        
    end
    return actor
end

function jitter_actor(actor, jitter)
    newactor = deepcopy(actor)
    for key in keys(newactor.energy)
        newactor.energy[key] += randn() * jitter
    end
    return newactor
end

function jitter_actor(actor, ctrl, states; jitter = 1.0)
    newactor = deepcopy(actor)
    for s in states
        state = collect(Tuple(s))
        if !ctrl.terminal_condition(state)
            for a in ctrl.action_space
                key = newactor.mapping(state, a)
                newactor.energy[key] += randn() * jitter
            end
        end
    end
    return newactor
end

"""
Test the effect of depth on the energy estimates

error(ξ - e_true) (depth)
"""