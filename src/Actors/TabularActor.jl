export TabularActor, init_tabular_actor_piql, update_function_piql, update_function_q, train!


mutable struct TabularActor{SA,F,G,A}
    energy::Dict{SA,Float64}
    visits::Dict{SA,Int}
    update::F  # (olde, visits, new) -> newavg  updates energies and visits according the the learning rule
    mapping::G # mapping(state, action) -> . key reducing size  of space and enforcing boundaries 
    action_space::A # for scanning for fall back.
    # mapping(state,action) -> key
    β::Float64
end


function update_function_q(;burnin = 1, p = 2/3)
    function update(olde, visits, new)
        return (new - olde) * (burnin / (burnin + visits))^p
    end 
    return update
end


function update_function_piql(;burnin = 1, p = 2/3, β = 1.0)
    function update(olde, visits, new)
        γ = (burnin / (burnin + visits))^p
        return -log1p(γ * expm1(-β * (new - olde))) / β
        # using the piql update rule, linear in the exponent
    end 
    return update
end

function example_init_ta(example_key::SA, update, mapping, action_space, β) where {SA}
    return TabularActor(
        Dict{SA,Float64}(), # energy
        Dict{SA,Int}(),     # visits
        update, mapping, action_space, β)
end

function init_tabular_actor_piql(ctrl; β = 1.0,
    mapping = (state,action) -> (state,action), update = update_function_piql(;burnin = 1, p = 2/3, β))
    example = mapping(ctrl.initial_state(), first(ctrl.action_space))
        # get type information about mapping result
    return example_init_ta(example, update, mapping, ctrl.action_space, β)
end

function init_tabular_actor_q(ctrl; β = 1.0,
    mapping = (state,action) -> (state,action), update = update_function_q(;burnin = 1, p = 2/3))
    example = mapping(ctrl.initial_state(), first(ctrl.action_space))
        # get type information about mapping result
    return example_init_ta(example, update, mapping, ctrl.action_space, β)
end


function (ta::TabularActor)(state,action)
    key = ta.mapping(state,action)
    if haskey(ta.energy,key)
        return ta.energy[key]
    else # fall back to (rough) free energy
        # could probably be done in a more elegant way
        # fairly agressive exploration
        out = 0.0
        nset = 0
        for a in ta.action_space
            key = ta.mapping(state,a)
            if haskey(ta.energy,key)
                out += exp(-ta.β * ta.energy[key])
                nset += 1
            end 
        end
        return ifelse(nset ==0, 0.0, -log(out/nset) / ta.β)
    end
end

"""
Destructive memory training, storing partition function
memory is composed of struct (at time of writing) consisting of

struct EnergyEstimate{S,A}
    state::S
    action::A
    β::Float64
    xi::Float64 # energy fluctuation realization
    logz::Float64
end 
"""
function train!(ta::TabularActor, memory)
    # for ee in memory
    # we only use the final value to to fair training comparisons
    ee = memory[end]
        key = ta.mapping(ee.state,ee.action)
        #  update visits
        if haskey(ta.visits,key)
            ta.visits[key] += 1
        else
            push!(ta.visits, key => 1)
        end
        # update energy
        if haskey(ta.energy,key)
            ta.energy[key] += ta.update(ta.energy[key],ta.visits[key], ee.xi)
        else
            push!(ta.energy, key => ee.xi)
        end
    #end
    resize!(memory,0) # remove everything, it's been used
end


##
#
# need to decide what the q learning rule is.
#
#