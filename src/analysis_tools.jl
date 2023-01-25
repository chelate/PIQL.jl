export performance_estimate


"""
Monte Carlo estimate for the performance of our state.
"""
function performance_estimate(ctrl, actor; state = ctrl.initial_state(), sa = initial_action(state, ctrl, actor))
    control_cost = 0.0
    state_cost = 0.0
    while true
        state_cost += sa.cost
        control_cost += sa.β * (sa.f - sa.u)
        if ctrl.terminal_condition(sa.state)
            break
        else
            sa = new_state_action(sa, ctrl, actor)
        end
    end
    return (state_cost, control_cost)
end

"""
S = -<F-E> = U - F
"""