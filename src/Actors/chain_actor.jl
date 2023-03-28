using SimpleChains
using Setfield
using ElasticArrays

"""
struct SelectionSample{S,A}
    sa1::StateAction{S,A}
    sa2::StateAction{S,A}
    ecrit1::Float64 #energy critic 1
    ecrit2::Float64 # energy critic 2
    β::Float64 # if possible always call β from cntrl problem
    λ::Float64 # λ at that time. measured in terms of the step time τ^-1.
end

must define a function 

function actor_grad(<:Actor,x, y)
    return Δparam gradient of parameters
end

https://pumasai.github.io/SimpleChains.jl/stable/examples/custom_loss_layer/

struct SampleContrast
    ΔEcrit::Float64
    ΔEact::Float64
    βeff::Float64
end


function make_xy(memory)
    x = reduce(hcat, flatten(sample) for sample in memory)
    y = [sample_contrast(sample) for sample in memory]
    return (x,y)
end

where flatten returns 
hcat(vcat(sa1.state, sa1.action, β),
vcat(sa2.state, sa2.action, β))
"""

mutable struct ChainActor{C,P,G,F,A}
    β::Float64 # β is a component of the actor.
    chain::C
    params::P
    grads::G
    prefun::F
    optimizer::A
    x::ElasticArray{Float32} # data buffer to reduce number of allocations
    y::Vector{SampleContrast}
    # a special object which encodes the target
    ndims::Int # number of input dimensions
end

function make_xy!(ca, batch)
    ndims = ca.ndims
    resize!(ca.x, ndims, length(batch))
    resize!(ca.y, length(batch))
    for (ii, sample) in enumerate(memory)
        (;sa1,sa2,β) = sample
        ca.y[ii] = sample_contrast(sample)
        z1 = ca.prefun(vcat(sa1.state, sa1.action, β))
        z2 = ca.prefun(vcat(sa2.state, sa2.action, β))
        for jj in eachindex(z1)
            ca.x[jj,2ii-1] = z1[jj]
            ca.x[jj,2ii] = z2[jj]
        end
    end
end


function train!(ca::ChainActor, memory; grad_steps = length(memory))
    #(x,y) = make_xy(memory; prefun = ca.prefun)
    make_xy!(ca, memory)
    chain_loss = SimpleChains.add_loss(ca.chain, ContrastiveCrossEntropyLoss(ca.y))
    SimpleChains.train_unbatched!(ca.grads, ca.params, chain_loss, ca.x, ca.optimizer, grad_steps);
end

function (a::ChainActor)(state, action)
    x = a.prefun(vcat(state, action, a.β)) # remove time in pendulum example
    # Very costly 50% of runtime
    return Float64(a.chain(x, a.params)[1])
end

function init_chain_actor(arch...; prefun = identity, ndims = 1)
    chain = SimpleChain(static(ndims), arch..., TurboDense(identity, 1)) # must terminate with a single dimension
    params = SimpleChains.init_params(chain)
    grads = similar(params)
    optimizer = SimpleChains.ADAM()
    ChainActor(1.0, 
        chain, 
        params, 
        grads, 
        prefun, 
        optimizer, 
        ElasticArray{Float32}(undef, ndims, 0), 
        Vector{SampleContrast}(), 
        ndims)
end




struct ContrastiveCrossEntropyLoss{T,Y<:AbstractVector{T}} <: SimpleChains.AbstractLoss{T}
    targets::Y
end

function calculate_loss(loss::ContrastiveCrossEntropyLoss, logits)
    # logits is an even number of outputs for the neural net
    y = loss.targets
    total_loss = zero(Float64)
    for ii in eachindex(y)
        Δε = logits[2*ii - 1] - logits[2*ii]
        total_loss += contrast_loss(y[ii],Δε)
    end
    total_loss
end

function SimpleChains.layer_output_size(::Val{T}, sl::ContrastiveCrossEntropyLoss, s::Tuple) where {T}
    SimpleChains._layer_output_size_no_temp(Val{T}(), sl, s)
end

function SimpleChains.forward_layer_output_size(::Val{T}, sl::ContrastiveCrossEntropyLoss, s) where {T}
    SimpleChains._layer_output_size_no_temp(Val{T}(), sl, s)
end


function (loss::ContrastiveCrossEntropyLoss)(previous_layer_output::AbstractArray, p::Ptr, pu)
    total_loss = calculate_loss(loss, previous_layer_output)
    total_loss, p, pu
end



function SimpleChains.chain_valgrad!(
    __,
    previous_layer_output::AbstractArray{T},
    layers::Tuple{ContrastiveCrossEntropyLoss},
    _::Ptr,
    pu::Ptr{UInt8},
) where {T}
    loss = getfield(layers, 1)
    total_loss = calculate_loss(loss, previous_layer_output)
    y = loss.targets
    # Store the backpropagated gradient in the previous_layer_output array.
    for i in eachindex(y)
        # Get the value of the last logit
        e1 = previous_layer_output[2*i-1]
        e2 = previous_layer_output[2*i]
        sign_arg = Float32(contrast_grad(y[i], e1-e2))
        previous_layer_output[2i-1] = - sign_arg
        previous_layer_output[2i] = sign_arg
    end
    return total_loss, previous_layer_output, pu
end
