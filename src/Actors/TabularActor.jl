export TabularActor, init_tabular_actor_piql, update_function_piql, update_function_q, train!


mutable struct TabularActor{SA,F,G,A}
    qtable::Dict{SA,Float64}
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
        return log1p(γ * expm1(β * (new - olde))) / β
        # using the piql update rule, linear in the exponent
    end 
    return update
end

function example_init_ta(example_key::SA, update, mapping, action_space, β) where {SA}
    return TabularActor(
        Dict{SA,Float64}(), # qtable
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
    if haskey(ta.qtable,key)
        return ta.qtable[key]
    else # fall back to (rough) free energy
        # could probably be done in a more elegant way
        # fairly agressive exploration
        out = 0.0
        nset = 0
        for a in ta.action_space
            key = ta.mapping(state,a)
            if haskey(ta.qtable,key)
                out += exp(ta.β * ta.qtable[key])
                nset += 1
            end 
        end
        return ifelse(nset ==0, 0.0, log(out/nset) / ta.β)
    end
end

"""
Destructive memory training, storing partition function
memory is composed of struct (at time of writing) consisting of

    struct QEstimate{S,A}
        sa::StateAction{S,A} # the state action information at this time step
        criticq::Float64 # from the state one step forward in time
        logz::Float64 # log z at state in sa.state
    end 

"""
function train!(ta::TabularActor, memory)
    #for qe in memory
    # we only use the final value to to fair training comparisons
    if length(memory) > 0
        qe = memory[end]
        key = ta.mapping(qe.sa0.state, qe.sa0.action)
        #  update visits
        if haskey(ta.visits,key)
            ta.visits[key] += 1
        else
            push!(ta.visits, key => 1)
        end
        # update energy
        if haskey(ta.qtable,key)
            ta.qtable[key] += ta.update(ta.qtable[key], ta.visits[key], qbound(qe))
        else
            push!(ta.qtable, key => qbound(qe))
        end
    # l = length(memory)
        resize!(memory,0) # remove everything, it's been used
        return 1
    else
        return 0
    end
    #end
end


##
#
# need to decide what the q learning rule is.
#
#