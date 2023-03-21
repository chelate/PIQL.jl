export get_ideal_actor, jitter_actor, get_reward, get_free_energy, excess_reward

"""
J_i = sum_a π_a e^-E_a^i
"""

function update_actor_everywhere!(actor, ctrl, states)
    discr = 0.0
    for ii in states
        state = collect(Tuple(ii))
        for a in ctrl.action_space
            oldq = actor(state,a)
            newq = criticq(state, a, ctrl, actor; critic_samples = 1)
            discr += abs(newq - oldq)
            key = actor.mapping(state,a)
            if haskey(actor.qtable, key)
                actor.qtable[key] = newq
            else
                push!(actor.qtable, key => newq)
            end
        end
    end
    return discr
end


function update_reward_everywhere!(reward_dict, actor, ctrl, states)
    # one step of the dynamic programming reccurence
    discr = 0.0
    for ii in states
        state = collect(Tuple(ii))
        if !ctrl.terminal_condition(state)
            Q = [actor(state, a) for a in ctrl.action_space]
            priors = [ctrl.action_prior(state, a) for a in ctrl.action_space]
            priors .*= 1/sum(priors)
            Qmax = maximum(Q)
            # z = sum( ctrl.action_prior(state, a) * exp(-actor.β * actor(state,a))
            #    for a in ctrl.action_space) / prior_normalization
            probs = priors .* exp.(actor.β .* (Q .- Qmax))
            probs = probs ./ sum(probs)
            R = sum(
                begin
                    state1 = ctrl.propagator(state,a)
                    p * (ctrl.reward_function(state, a, state1) - log(p/q) + ctrl.γ * reward_dict[state1])
                end
                for (p,q,a) in zip(probs, priors, ctrl.action_space))
            discr += abs(R - reward_dict[state])
            reward_dict[state] = R
        end
    end
    return discr
end


function get_reward(ctrl, actor, states; maxiter = 1e5)
    # Estimate the reward-to-go of the current policy as a function of state
    reward_dict = Dict(collect(Tuple(state)) => 0.0 for state in states)
    iter  = 0
    while true
        iter +=1
        discr = update_reward_everywhere!(reward_dict, actor, ctrl, states)
        if (discr < 1e-9) | (iter > maxiter) 
            break
        end
    end
    return reward_dict
end

function get_value_function(ctrl, actor, states)
    # Estimate the reward-to-go of the current policy as a function of state
    return Dict(collect(Tuple(state)) => value_function(collect(Tuple(state)), ctrl, actor) for state in states)
end

function excess_reward(ctrl, actor0, actor1, states)
    # actor 0 is the idealized optimal actor
    fd = get_value_function(ctrl, actor0, states)
    cd = get_reward(ctrl, actor1, states)
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
    for key in keys(newactor.qtable)
        newactor.qtable[key] += randn() * jitter
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
                newactor.qtable[key] += randn() * jitter
            end
        end
    end
    return newactor
end

"""
Test the effect of depth on the qtable estimates

error(ξ - e_true) (depth)
"""