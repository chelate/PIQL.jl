import Random: shuffle

struct Growable{S,F}
    dict::Dict{S,Float64}
    default::F # function that maps from state to a default value
end

function Growable(s::S ; default::F = x -> 0.0) where {S,F}
    Growable{S,F}(Dict(s => 0.0),default)
end

function (g::Growable)(i) # the struct is callable, and grows automatically
    # we expect it to get called in the recursion relations.
    if !haskey(g.dict,i)
        push!(g.dict, i => g.default(i))
    end 
    g.dict[i]
end


# Pure value iteration


function update_z!(Z, ctrl, V; β = 1.0)
    diff = 0.0 # running difference
    old_len = length(Z.dict) 
    for s0 in shuffle(collect(keys(V.dict)))
        if ctrl.terminal_condition(s0)
            new = exp(0.0 - V(s0))
        else
            new = z_recursion2(ctrl, 
                (s,a) -> controlV(ctrl, V, s, a), # V-determined Q-funciton
                    V, s0, Z; β)
        end
        diff += abs(log(new) - log(Z(s0)))
        η = 0.5
        Z.dict[s0] = η*new + (1-η)*Z(s0)
    end
    return diff + length(Z.dict) - old_len # function change plus any new states visited.
end

function generate_z(ctrl, V; β = 1.0)
    s = ctrl.initial_state()
    Z = Growable(s, default = x -> 1.0)
    while true
        diff = update_z!(Z, ctrl, V; β)
        if diff  < 10^(-9)
            break
        end
    end
    return Z
end

function z_updateV(ctrl, V; β = 1.0)
    Z = generate_z(ctrl, V; β)
    out_dict = Dict(k => V(k) + 
        log(Z(k))/β
        for k in keys(V.dict))
    return Growable(out_dict, V.default)
end

function update_ν!(ν, ctrl)
    diff = 0.0 # running difference
    old_len = length(ν.dict)
    for s0 in shuffle(collect(keys(ν.dict)))
        if ctrl.terminal_condition(s0)
            new = 0.0
        else
            new = truevalue_recursion(ctrl, s0, ν)
        end
        diff += abs(new - ν(s0))
        ν.dict[s0] = new
    end
    return diff + length(ν.dict) - old_len
end

function generate_ν(ctrl)
    s = ctrl.initial_state()
    ν = Growable(s, default = x->0.0)
    while true
        diff = update_ν!(ν, ctrl)
        if diff  < 10^(-12)
            break
        end
    end
    return ν
end

function update_logz!(logz, ctrl, V; β = 1.0)
    diff = 0.0 # running difference
    old_len = length(logz.dict) 
    for s0 in shuffle(collect(keys(V.dict)))
        if ctrl.terminal_condition(s0)
            new = 0.0 - V(s0)
        else
            new = logz_recursion(ctrl, 
                (s,a) -> controlV(ctrl, V, s, a), # V-determined Q-funciton
                    V, s0, logz; β)
        end
        diff += abs(new - logz(s0))
        η = 0.5
        logz.dict[s0] = logaddexp(log(η) + new, log(1-η)+logz(s0))
    end
    return diff + length(logz.dict) - old_len # function change plus any new states visited.
end

function generate_logz(ctrl, V; β = 1.0)
    s = ctrl.initial_state()
    logz = Growable(s, default = x -> 0.0)
    while true
        diff = update_logz!(logz, ctrl, V; β)
        if diff  < 10^(-12)
            break
        end
    end
    return logz
end

function logz_updateV(ctrl, V; β = 1.0)
    logz = generate_logz(ctrl, V; β)
    out_dict = Dict(k => V(k) + logz(k)/β
        for k in keys(V.dict))
    return Growable(out_dict, V.default)
end


### Optimality gap

function update_optimality_gap!(gap, ctrl, V; ν = generate_ν(ctrl))
    diff = 0.0 # running difference
    old_len = length(gap.dict) 
    for s0 in shuffle(collect(keys(V.dict)))
        if ctrl.terminal_condition(s0)
            new = 0.0
        else
            new = optimality_gap_value_recursion(ctrl, V, ν, s0, gap)
        end
        diff += abs(new - gap(s0))
        gap.dict[s0] = new
    end
    return diff + length(gap.dict) - old_len # function change plus any new states visited.
end

function generate_optimality_gap(ctrl, V; ν = generate_ν(ctrl))
    s = ctrl.initial_state()
    gap = Growable(s, default = x -> 0.0)
    while true
        diff = update_optimality_gap!(gap, ctrl, V; ν)
        if diff  < 10^(-12)
            break
        end
    end
    return gap
end

function optimality_gap_value_recursion(ctrl, V, ν, s0, gap)
    optimality_gap_recursion(ctrl, (s,a) -> controlV(ctrl, V, s, a), ν, s0, gap)
end

function optimality_gap_recursion(ctrl, Q, ν, s0, gap)
    out = 0.0
    for a in ctrl.action_space
        out += action_probability(ctrl, Q, s0, a) * (
            controlQ(ctrl, Q, s0, a) - controlV(ctrl, ν, s0, a) + 
            ctrl.propagator_average(s0, a, gap)
        )
    end
    out
end