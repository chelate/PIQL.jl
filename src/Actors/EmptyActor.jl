"""
We define an actor interface: an actor is a callable struct

struct GenericActor
    parameters

end

function (a::Actor)(state,action)
    ...
    return E_actor
end

function loss(actor, minibatch)
    ...
    return E_actor
end
"""
function empty_actor(;β = 1.0)
    EmptyActor(β)
end


mutable struct EmptyActor
    β::Float64
end

function (actor::EmptyActor)(s,a) # default actor for just running simulations
    0.0
end

