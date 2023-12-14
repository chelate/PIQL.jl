using PIQL
using Test

s = (20,20)
gw = make_gridworld(s; density = 0.1)
gwctrl = make_ctrl(gw; randomness = 0.1)
initial_value_dict = Dict(s => 0.0 for s in state_iterator(gw))


bbctrl = binary_bandit_problem(rounds = 20, ncoins = 2, prior = (1,1), payoff = 1.0, γ = 1.0)

@testset "PIQL.jl" begin
    @test gw.walls[gw.goal] == false
    @test gw.walls[gwctrl.initial_state()] == false

    i0 = bbctrl.initial_state()
    i1  = bbctrl.propagator(i0,1)
    @test i0.rounds - i1.rounds == 1
    @test average_reward(bbctrl,i0, 1) == 1/2 # average reward is half payoff
    @test average_reward(bbctrl,i1, 1) == i1.belief[1][1]/sum(i1.belief[1]) # average reward is half payoff
end
