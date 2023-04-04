using Flux
using Zygote

"""
mutable struct PVActor{P,V}
    policy::P # state,action -> η mapping
    value::V # state -> V
    β::Float64
end

function (pv::PVActor)(state, action)
    v = pv.value(state)
    η = pv.policy(state,action)
    return η / pv.β + v # the Q
end

function train!(pv::PVActor, memory)
    #for qe in memory
    # we only use the final value to to fair training comparisons
    if length(memory) > 0
        qe = memory[end]
        train!(pv.value, qe)
        train!(pv.policy, qe)
    #end
    #l = length(memory)
    resize!(memory,0) # remove everything, it's been used
        return 1
    else
        return 0
    end
end
"""


mutable struct ChainValue{C,P,G,F,A}
    chain::C
    params::P
    grads::G
    prefun::F
    optimizer::A
    x::ElasticArray{Float32} # data buffer to reduce number of allocations
    # includes β
    y::Vector{Y}
    # a special object which encodes the target
    indims::Int # number of input dimensions

    # usually 
end


struct DeepValue{C}
    chain::C
    # memory
    memsize::Int64
    x::ElasticArray{Float32}
    v::Vector{Float32}
    # 

end

train!(ch::Chain, qe)


chain = Chain(
  Dense(24, 8, tanh; bias = true),
  Flux.Dropout(0.2),
  Dense(8, 2, identity; bias = false)
);
chain.layers[2].active = true # activate dropout

ya = Array(y);

@benchmark gradient(Flux.params($chain)) do
  Flux.mse($chain($x), $ya)
end