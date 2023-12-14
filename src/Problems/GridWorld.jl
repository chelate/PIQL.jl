
export make_gridworld, make_ctrl
# struct ControlProblem{A, U, P, R, PA, T, W}
#     action_space::Vector{A} # something that we can iterate over
#     action_prior::U # π(s,a) -> Float64 exactly like energy
#     propagator::P # p(x0, a) -> x1 ("random" state)
#     reward_function::R # r(x0, a, x1) -> reward ::Float64
#     # given in entropic units already
#     propagator_average::PA # (s,a,f) -> K·f
#     terminal_condition::T # T(x) -> bdol
#     initial_state::W # W() -> x0 generates inital states of interest
#     γ::Float64 # positive number less than one discount over time
# end

using StatsBase

"""
Actions are indexed by natural numbers, 0-3 in 2D
"""
function step_choices(; dim::Val{G} = Val(2)) where G
     0:2*G-1
end

function step(a; dim::Val{G} = Val(2)) where G
    CartesianIndex(ntuple(ii -> (1-2*mod(a,2))*(div(a,2) == ii-1) , G))
end


function take_step(x0::G, a; dim = Val(2), walls = Bool[]) where G
    x1::G = x0 + step(a;dim)
    return ifelse(walls[x1], x0, x1)
end


struct Gridworld
    walls::Array{Bool}
    goal::CartesianIndex
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
            return CartesianIndices(walls)[ii]
        end
    end
end

function get_reachable(walls, goal)
    reachable_set = Set([goal])
    boundary = Set([goal])
    dim = Val(ndims(walls))
    while !isempty(boundary)
        union!(reachable_set, boundary)
        boundary = reduce(union,
            Set(ii + s
                for s in step.(step_choices(; dim); dim) if 
                    !in(ii + s, reachable_set) & !walls[(ii + s)] )
            for ii in boundary)
    end
    return reachable_set
end

function make_gridworld(size; density = 0.1)
    walls = make_walls(size; density)
    goal = draw_not_wall(walls)
    reachable_set = get_reachable(walls, goal)
    for ii in CartesianIndices(walls)
        if !in(ii,reachable_set)
             walls[ii] = true
         end
    end
    return Gridworld(walls,goal)
end

function make_ctrl(gw::Gridworld; 
    randomness = 0.1, # likelihood of choosing a random action
    reward_scale = 0.2, # temperature
    step_cost = 0.1*reward_scale, # entropic cost of a step
    reward = 1*reward_scale, # reward of reaching the end
    γ = 0.99) # discounting rate
    let dim = Val(ndims(gw.walls)), len = length(step_choices(;dim)), walls = gw.walls
        # attempt to make the closure fast
    function pa(s,a,f) # propagator average
        out::Float64 = 0.0
        for aa in step_choices(;dim)
            out += ((a == aa)*(1-randomness) + (randomness / len)) * f(take_step(s,aa; dim, walls))
        end
        out 
    end
    ControlProblem(
        step_choices(;dim),
        (s,a)->1.0 / len,
        (s,a) -> take_step(s,
            ifelse(rand() < randomness, sample(step_choices(;dim)),a); 
            dim, walls),
        (s0,a,s1) -> ifelse(s1 == gw.goal, reward, - step_cost),
        pa,
        s -> s == gw.goal ,
        () -> draw_not_wall(gw.walls),
        γ) end
end

state_iterator(gw::Gridworld) = filter(ii -> !gw.walls[ii] & (ii != gw.goal),
            CartesianIndices(gw.walls))