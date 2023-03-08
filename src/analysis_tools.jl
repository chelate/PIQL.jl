export performance_estimate


"""
Monte Carlo estimate for the performance of our state.
"""
function performance_estimate(ctrl, actor; state = ctrl.initial_state(), sa = initial_action(state, ctrl, actor))
    control_cost = 0.0
    state_cost = 0.0
    r = 1.0 # discount factor decays forward in time with gamma
    while true
        state_cost += r * sa.cost
        control_cost += r * (sa.f - sa.u)
        if ctrl.terminal_condition(sa.state)
            break
        else
            sa = new_state_action(sa, ctrl, actor)
        end
        r = ctrl.γ * r
    end
    return (state_cost, control_cost)
end

function softq_estimate(ctrl, actor; 
    state = ctrl.initial_state(), sa = initial_action(state, ctrl, actor), depth = 1)
    control_cost = 0.0
    state_cost = 0.0
    r = 1.0 # discount factor
    for _ in 1:depth
        control_cost += r *(sa.f - sa.u)
        state_cost += r * sa.cost
        if ctrl.terminal_condition(sa.state)
            break
        else
            sa = new_state_action(sa, ctrl, actor)
        end
        r = ctrl.γ * r
    end
    state_cost += r * sa.cost
    value_function = sa.f
    return (state_cost + control_cost + value_function)
end

"""
S = -<F-E> = U - F

but KL divergence is F - U.
"""