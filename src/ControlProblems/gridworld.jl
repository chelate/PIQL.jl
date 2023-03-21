#include("agent.jl")
export Gridworld, make_ctrl, make_gridworld, state_iterator

"""
Just a refernce struct 
struct ControlProblem{A, U, P, C, T, W}
    action_space::Vector{A} # something that we can iterate over
    action_prior::U # π(s,a) -> Float64 exactly like energy
    propagator::P # p(x0, a) -> x1 ("random" state)
    cost_function::C # c(x0, a, x1) -> Cost ::Float64
    terminal_condition::T # T(x) -> bool
    initial_state::W # W() -> x0 generates inital states of interest
    γ::Float64 # positive number discount over time
end
"""


"""
Actions are indexed by natural numbers
"""
step_choices(;dim = 2) = collect(0:2*dim-1)
function step(a; dim = 2)
    [(1-2*mod(a,2))*(div(a,2) == ii) for ii in 0:dim-1]
end


function take_step(x0, a; dim = 2, walls = Bool[])
    x1 = x0 .+ step(a;dim)
    return ifelse(walls[x1...], x0, x0 .+ step(a;dim))
end

terminal_condition(x; dim = 2, term = ones(dim)) = (x .== term) 


struct Gridworld
    walls::Array{Bool}
    goal::Vector{Int}
end

function make_walls(size; density = 0.1)
    walls = falses((size .+ 2)...)
    walls[begin,:] .= true
    walls[end,:] .= true
    walls[:,begin] .= true
    walls[:,end] .= true
    for ii in eachindex(walls)
        if rand() < density
            walls[ii] = true # put down random barriers
        end
    end
    return walls
end

function draw_not_wall(walls)
    while true
        ii = rand(LinearIndices(walls))
        if !walls[ii]
            return collect(Tuple(CartesianIndices(walls)[ii]))
        end
    end
end

function get_reachable(walls, goal)
    reachable_set = Set([goal])
    boundary = Set([goal])
    D = ndims(walls)
    while !isempty(boundary)
        union!(reachable_set, boundary)
        boundary = reduce(union,
            Set(ii .+ s
                for s in step.(step_choices(;dim = D);dim = D) if 
                    !in(ii .+ s, reachable_set) & !walls[(ii .+ s)...] )
            for ii in boundary)
    end
    return reachable_set
end

function make_gridworld(size; density = 0.1)
    walls = make_walls(size; density)
    goal = draw_not_wall(walls)
    reachable_set = get_reachable(walls, goal)
    for ii in CartesianIndices(walls)
        if !in(collect(Tuple(ii)),reachable_set)
             walls[ii] = true
         end
    end
    return Gridworld(walls,goal)
end

function make_ctrl(gw::Gridworld; step_cost = 1, reward = 5, γ = 0.99)
    dim = ndims(gw.walls)
    ControlProblem(
        step_choices(;dim),
        (s,a)->1.0,
        (s,a) -> take_step(s,a; dim, gw.walls),
        (s0,a,s1) -> ifelse(s1 == gw.goal, reward, - step_cost),
        s -> s == gw.goal ,
        () -> draw_not_wall(gw.walls),
        γ)
end

state_iterator(gw::Gridworld) = filter(ii -> !gw.walls[ii],
            CartesianIndices(gw.walls))