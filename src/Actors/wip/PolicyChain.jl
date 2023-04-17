
struct KLDivTarget
    pivec::Vector{Float32}
    eta_t::Float32
    criticeta::Float32
    aii::Int64
end

struct KLDivLoss{T,Y<:AbstractVector{T}} <: SimpleChains.AbstractLoss{T}
    target::Y 
end

target(loss::KLDivLoss) = loss.targets
(loss::KLDivLoss)(::Int) = loss

function calculate_loss(losslayer::KLDivLoss, hmat)
    y = losslayer.targets
    total_loss = zero(eltype(hmat))
    for ii in eachindex(y)
        total += loss(y[ii], view(hmat, :, ii))
    end
    total_loss
end

function loss(a::KLDivTarget, hvec)
    (   ;pivec, # prior vector (normalized)
    eta_t, # eta when the action was drawn
    criticeta,
    aii) = a
    Z = sum(pivec[ii] * exp(hvec[ii]) for ii in eachindex(hvec))
    ((hvec[aii] - log(Z)) - criticeta) * exp(hvec[aii] - eta_t) / Z
end


function stochastic_grad!(
    hvec, y::KLDivTarget) # action index )
    (   ;pivec, # prior vector (normalized)
        eta_t, # eta when the action was drawn
        criticeta,
        aii) = y
    Z = sum(pivec[ii] * exp(hvec[ii]) for ii in eachindex(hvec))
    change = - (criticeta - (hvec[aii] - log(Z))) * exp(hvec[aii] - eta_t) / Z
    for ii in eachindex(y)
        hvec[ii] = ((ii == aii) - (pivec[ii] * exp(hvec[ii]) / Z)) * change
    end
    return change # expectation of change is the loss
end

# break up into target and loss layer

mutable struct ChainPolicy{C,P,G,F,A,B}
    β::Float64 # β is a component of the actor.
    chain::C
    params::P
    grads::G
    prefun::F # pretransormation
    optimizer::A
    x::ElasticArray{Float32} # data buffer to reduce number of allocations
    y::Vector{KLDivLoss}
    # a special object which encodes the target
    ndims::Int # number of input dimensions
    action_index::B
end

function (a::ChainPolicy)(state, action)
    πvec = [prior(state,action) for a in a.action_space]
    πvec *= inv(sum(πvec))
    ii = a.action_index(action)
    wvec = a.chain(prefun(state), a.params)
    return wvec[ii] - log(sum(πvec .* exp.(wvec)))
end

# to do, define loss action

function (loss::KLDivLoss)(previous_layer_output::AbstractArray, p::Ptr, pu)
    total_loss = calculate_loss(loss, previous_layer_output)
    total_loss, p, pu
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

# define valgrad