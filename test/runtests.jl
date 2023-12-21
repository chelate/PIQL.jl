using PIQL
using Test

s = (10,10)
gw = make_gridworld(s; density = 0.1)
gwctrl = gridworld_ctrl(gw; randomness = 0.1, γ = 0.99, reward_scale = 1.0, reward = 2.0)
weaker_gwctrl = modify(gwctrl, reward_function = (s,a,ss) -> 0.8*gwctrl.reward_function(s,a,ss) ) # softer control
stronger_gwctrl = modify(gwctrl, reward_function = (s,a,ss) -> 1.2*gwctrl.reward_function(s,a,ss) ) # harder 

ν = generate_ν(gwctrl)
V_lo = generate_ν(weaker_gwctrl)
V_hi = generate_ν(stronger_gwctrl)



vhi_1 = logz_updateV(gwctrl, V_hi; β = 1.0)
vlo_1 = logz_updateV(gwctrl, V_lo; β = 1.0)

vhi_0 = logz_updateV(gwctrl, V_hi; β = 2*10^(-3))
vlo_0 = logz_updateV(gwctrl, V_lo; β = 2*10^(-3))



bbctrl = binary_bandit_problem(rounds = 20, ncoins = 2, prior = (1,1), payoff = 1.0, γ = 1.0)

@testset "PIQL.jl" begin
    # gridworld
    @test gw.walls[gw.goal] == false
    @test gw.walls[gwctrl.initial_state()] == false

    wiggle = 10^(-2)
    # Test the inequalities
    @test all( vlo_1(k) >= ν(k) - wiggle for k in keys(ν.dict))
    @test all( vhi_1(k) >= ν(k) - wiggle for k in keys(ν.dict))
    @test all( vlo_0(k) <= ν(k) + wiggle for k in keys(ν.dict))
    # not sure why this one alone is failing
    @test all( vhi_0(k) <= ν(k) + wiggle for k in keys(ν.dict))
    # gridworld + tabular solver

    i0 = bbctrl.initial_state()
    i1  = bbctrl.propagator(i0,1)
    @test i0.rounds - i1.rounds == 1
    @test average_reward(bbctrl,i0, 1) == 1/2 # average reward is half payoff
    @test average_reward(bbctrl,i1, 1) == i1.belief[1][1]/sum(i1.belief[1])

end
