export get_ideal_actor, jitter_actor, get_cost, get_free_energy, excess_cost

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


function update_cost_everywhere!(cost_dict, actor, ctrl, states)
    # one step of the dynamic programming reccurence
    discr = 0.0
    for ii in states
        state = collect(Tuple(ii))
        if !ctrl.terminal_condition(state)
            energies = [actor(state, a) for a in ctrl.action_space]
            priors=[ctrl.action_prior(state, a) for a in ctrl.action_space]
            prior_normalization=sum(priors)
            emin=minimum(energies)
            # z = sum( ctrl.action_prior(state, a) * exp(-actor.β * actor(state,a))
            #    for a in ctrl.action_space) / prior_normalization
            probs = priors.* exp.(-actor.β .*(energies .-emin)) ./ prior_normalization
            probs = probs ./ sum(probs)
            R = sum(
                begin
                    state1 = ctrl.propagator(state,a)
                    p * (ctrl.cost_function(state, a, state1) + log(prior_normalization*p/q) + ctrl.γ *cost_dict[state1])
                end
                for (p,q,a) in zip(probs, priors, ctrl.action_space))
            discr += abs(R - cost_dict[state])
            cost_dict[state] = R
        end
    end
    return discr
end


function get_cost(ctrl, actor, states; maxiter = 1e5)
    # Estimate the cost-to-go of the current policy as a function of state
    cost_dict = Dict(collect(Tuple(state)) => 0.0 for state in states)
    iter  = 0
    while true
        iter +=1
        discr = update_cost_everywhere!(cost_dict, actor, ctrl, states)
        if (discr < 1e-9) | (iter > maxiter) 
            break
        end
    end
    return cost_dict
end

function get_free_energy(ctrl, actor, states; maxiter = 1e5)
    # Estimate the cost-to-go of the current policy as a function of state
    return Dict(collect(Tuple(state)) => free_energy(collect(Tuple(state)), ctrl, actor) for state in states)
end

function excess_cost(ctrl, actor0, actor1, states)
    # actor 0 is the idealized optimal actor
    fd = get_free_energy(ctrl, actor0, states)
    cd = get_cost(ctrl, actor1, states)
    mean(cd[st] - val for (st,val) in pairs(fd))
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