

function performance_estimate(ctrl, actor; n_simulations = 100, state = ctrl.initial_state())
    control_cost = 0.0
    state_cost = 0.0
    sa = initial_action(state, ctrl, actor)
    while !ctrl.terminal_condition(sa.state)
        state_cost += sa.cost
        control_cost += sa.β * (sa.u - sa.f)
        sa = new_state_action(sa, ctrl, actor)
    end
    return (state_cost, control_cost)
end