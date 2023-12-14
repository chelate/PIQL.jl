

# Pure value iteration


function update_z!(z_dict, ctrl, v_dict; β = 1.0)
    diff = 0.0 # running difference
    for s0 in shuffle(keys(z_dict))
        if ctrl.terminal_condition(s0)
            new = exp(0.0 - v_dict[s0])
        else
            new = z_recursion(ctrl, 
                (s,a) -> controlV(ctrl, s -> vdict[s], s, a), # V-determined Q-funciton
                    s -> v_dict[s], s0, s -> z_dict[s]; β)
        end
        diff += (log(new) - log(Z[s0]))^2
        Z[s0] = new
    end
end

function generate_z(ctrl, v_dict)
    z_dict = Dict(k => 1.0 for k in keys(v_dict))
    while true
        diff = update_z(z_dict, ctrl, v_dict)
        if diff  < 10^(-5)
            break
        end
    end
    return z_dict
end

function update_ν!(ν_dict, ctrl)
    diff = 0.0 # running difference
    for s0 in shuffle(keys(z_dict))
        if ctrl.terminal_condition(s0)
            new = 0.0
        else
            new = truevalue_recursion(ctrl, s, s -> ν_dict[s])
        end
        diff += (log(new) - log(ν_dict[s0]))^2
        Z[s0] = new
    end
end

function generate_ν(ctrl, v_dict)
    z0 = Dict(k => 1.0 for k in keys(v_dict))
    while true
        diff = update_ν(ν_dict, ctrl)
        if diff  < 10^(-5)
            break
        end
    end
    return z0
end