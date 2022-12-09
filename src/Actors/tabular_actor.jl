
mutable struct Tabular_actor
    energy::Dict{Tuple{Int,Int},Float64}
    visits::Dict{Tuple{Int,Int},Int}
    update::Function  ## updates energies and visits according the the learning rule
    hyperparameters::Dict{String,Float64}
end


function (a::Tabular_actor)(state,action)
    a.energy[(state,action)]
end


#


function initilize_tabular_dict(states,actions,value)
    a= Dict{Tuple{Int,Int},Float64}
    merge!(a, Dict((1,1)=> 1))
    for ii in states
        for jj in actions
            a=merge!(a, Dict((ii,jj)=> value))
        end
    end
    return a
end


function make_softQL_actor(gridworld::Gridworld,actor_hyper::Dict{String,Float64})
    function softQL_update(actor::Tabular_actor,s,a,g)

        actor.visits[ (s,a)] += 1
        alpha_q = actor.visits[ (s,a)] ^(-actor.hyperparameters["omega"])
        actor.energy[ (s,a)]  -= alpha_q* g

    end


     return Tabular_actor(
        initilize_tabular_dict(1:length(gridworld.grid),(gridworld.actions),0.0),
        initilize_tabular_dict(1:length(gridworld.grid),(gridworld.actions),0),
        softQL_update,
        actor_hyper
        )
end
