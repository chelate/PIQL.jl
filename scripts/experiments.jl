using PIQL
using JLD2

begin
    s = (18,18)
    gw = make_gridworld(s; density = 0.2)
end
     

function learning_traj(gwctrl, v0; β = 1.0, n = 20)
    out = [v0]
    vnew = v0
    for _ = 1:n
        vnew = logz_updateV(gwctrl, vnew; β)
        push!(out,vnew)
    end
    return out
end

function run_experiment(gw; randomness = 0.01, γ = .99, 
    starting_temp = 0.1,
    βrange = exp.(range(-6,0,8)))
    #generates a named tuple
    gwctrl = gridworld_ctrl(gw; randomness, γ)

    weaker_gwctrl = modify(gwctrl, reward_function = (s,a,ss) -> starting_temp*gwctrl.reward_function(s,a,ss) )
    
    ν = generate_ν(gwctrl)
    (_,furthest) = findmin(ν.dict)
    V_lo = generate_ν(weaker_gwctrl)
    experiment = Dict(β => learning_traj(gwctrl, V_lo; β, n = 5)    
        for β in βrange)
    optimality_gap = Dict(β => 
            [generate_optimality_gap(gwctrl, v; ν) for v in vec]
        for (β,vec) in pairs(experiment))

    return (ν = ν, traj = experiment, gap = optimality_gap,
        randomness = randomness, γ = γ, ctrl = gwctrl, s0 = furthest)
end

gap = generate_optimality_gap(out.ctrl, first(values(out.traj))[1])
extrema(matrixify(gap))


out = run_experiment(gw)