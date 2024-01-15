using JLD2
using Distributed
addprocs() 
@everywhere using PIQL
@everywhere function learning_traj(gwctrl, v0; β = 1.0, n = 20)
    out = [v0]
    vnew = v0
    for _ = 1:n
        vnew = logz_updateV(gwctrl, vnew; β)
        push!(out,vnew)
    end
    return out
end

@everywhere function run_experiment_distributed(gw; randomness = 0.01, γ = .99, 
    starting_temp = 0.1,
    βrange = exp.(range(-6,0,8)))
    #generates a named tuple
    gwctrl = gridworld_ctrl(gw; randomness, γ)

    weaker_gwctrl = modify(gwctrl, reward_function = (s,a,ss) -> starting_temp*gwctrl.reward_function(s,a,ss) )
    
    ν = generate_ν(gwctrl)
    (_,furthest) = findmin(ν.dict)
    V_lo = generate_ν(weaker_gwctrl)
    experiment = Dict(pmap(β -> β => learning_traj(gwctrl, V_lo; β, n = 5), βrange)...)
    optimality_gap = Dict(β => 
            pmap(v -> generate_optimality_gap(gwctrl, v; ν), vec)
        for (β,vec) in pairs(experiment))

    return (ν = ν, traj = experiment, gap = optimality_gap,
        randomness = randomness, γ = γ, ctrl = gwctrl, s0 = furthest)
end


begin
s = (18,18)
gw = make_gridworld(s; density = 0.2)
end
 

begin
r0_05g0_95 = @spawn run_experiment_distributed(gw, randomness = 0.05, γ = .95)
r0_01g0_99 = @spawn run_experiment_distributed(gw, randomness = 0.01, γ = .99)
r0_00g0_99 = @spawn run_experiment_distributed(gw, randomness = 0.00, γ = .99)
r0_00g0_9999 = @spawn run_experiment_distributed(gw, randomness = 0.00, γ = 0.9999)
end

out = fetch(r0_01g0_99)
@save "data/r0_01g0_99.jld2" out

out = fetch(r0_05g0_95)
@save "data/r0_05g0_95.jld2" out

out = fetch(r0_00g0_99);
@save "data/r0_00g0_99.jld2" out

out = fetch(r0_00g0_9999);
@save "data/r0_00g0_9999.jld2" out
