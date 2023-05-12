export init_chainpv

using SimpleChains
using ElasticArrays
using LogExpFunctions



"""
ChainPV
the inputs are

[prefun(state)] = 

the outputs are all unitless, 

[ action H] + [ β V ]
so note
 (η + v) / β = Q
"""

struct CombinedDivLoss{T,Y<:AbstractVector{T}} <: SimpleChains.AbstractLoss{T}
    targets::Y
end
# loss defined in terms of an inner function 
function (loss::CombinedDivLoss)(previous_layer_output::AbstractArray{T}, p::Ptr, pu) where {T}
    total_loss = calculate_loss(loss, previous_layer_output)
    total_loss, p, pu
end
# define the target
SimpleChains.target(loss::CombinedDivLoss) = loss.targets

# not sure what this does, looks like a constructor method??
(loss::CombinedDivLoss)(x::AbstractArray) = CombinedDivLoss(x)

# preallocation control
function SimpleChains.layer_output_size(::Val{T}, sl::CombinedDivLoss, s::Tuple) where {T}
    SimpleChains._layer_output_size_no_temp(Val{T}(), sl, s)
end
function SimpleChains.forward_layer_output_size(::Val{T}, sl::CombinedDivLoss, s) where {T}
    SimpleChains._layer_output_size_no_temp(Val{T}(), sl, s)
end

function calculate_loss(loss::CombinedDivLoss, logits)
    y = loss.targets
    total_loss = zero(eltype(logits))
    logZZ = logsumexp(yy.logZ_t for yy in y) # normalizing constant for the Z
    logZV = logsumexp( y[i].β*(logits[i,end] - y[i].V_t) for i in eachindex(y)) 
    isnan(logZZ) && error("logZZ pooped first")
    isnan(logZV) && error("logZV pooped first $(length(y))")
    # normalizing constant for the value functions
    for i in eachindex(y)
        p_i = view(logits, :, i)
        y_i = y[i]
        total_loss += combined_loss(p_i, y_i; logZZ, logZV)

    end
    total_loss
end

function SimpleChains.chain_valgrad!(
    __,
    previous_layer_output::AbstractArray{T},
    layers::Tuple{CombinedDivLoss},
    _::Ptr,
    pu::Ptr{UInt8},
) where {T}
    loss = getfield(layers, 1)
    total_loss = 0.0
    y = loss.targets
    logZZ = logsumexp(yy.logZ_t for yy in y)
    logZV = logsumexp( y[i].β*(previous_layer_output[i,end] - y[i].V_t) 
        for i  in eachindex(y))
    isnan(logZZ) && error("logZZ pooped first")
    isnan(logZV) && error("logZV pooped first 
        $(print(y)) 
        $(print(previous_layer_output[:,end]))")
    # Store the backpropagated gradient in the previous_layer_output array.
    for i in eachindex(y)
        p_i = view(previous_layer_output, :, i)
        y_i = y[i]
        total_loss += setgrad!(p_i, y_i; logZZ, logZV)
    end
    return total_loss, previous_layer_output, pu
end

# inner methods

struct CombinedDivTarget
    pivec::Vector{Float32}
    eta_t::Float32
    criticeta::Float32
    actioni::Int64 # action index
    V_t::Float32 # value function for agent
    logZ_t::Float32 # back-propagated logZ_t
    β::Float32 # beta for the acting agent
end

front(itr, n=1) = Iterators.take(itr, length(itr)-n)
# want to append value to the end of the policy vector

function combined_loss(hvec, y::CombinedDivTarget; logZZ = 0.0, logZV = 0.0)
    # the sum of policy and value losses
    # strict subset of computations of setgrad
    # hopefully never called
        ( ;pivec, # prior vector (normalized)
        eta_t, # eta when the action was drawn
        criticeta,
        actioni, V_t, logZ_t, β) = y
    # value function
    loss = - exp(logZ_t - logZZ) * (hvec[end] - β*V_t - logZV)
    Z = sum(pivec[ii] * exp(p[ii]) for ii in front(eachindex(hvec)))
    change = - (criticeta - (hvec[actioni] - log(Z))) * exp(hvec[actioni] - eta_t) / Z
    return loss + change
end

function setgrad!(hvec, y::CombinedDivTarget; logZZ = 0.0, logZV = 0.0)
    # hvec is the previous layer_
    # y is a specialized type that carries all neccesary gradient information
    # the sum of policy and value losses
    ( ;pivec, # prior vector (normalized)
        eta_t, # eta when the action was drawn
        criticeta,
        actioni, V_t, logZ_t, β) = y
    # value function
    v = hvec[end]
    loss = - exp(logZ_t - logZZ) * (v - β*V_t - logZV)
    dv = exp(v - β*V_t - logZV) - exp(logZ_t - logZZ) 
    hvec[end] = dv
    # policy function
    Z = sum(pivec[ii] * exp(hvec[ii]) for ii in front(eachindex(hvec)))
    change = - (criticeta - (hvec[actioni] - log(Z))) * exp(hvec[actioni] - eta_t) / Z
    isnan(change) && error("change pooped first")
    for ii in front(eachindex(hvec))
        hvec[ii] = ((ii == actioni) - 
            (pivec[ii] * exp(hvec[ii]) / Z)) * change # policy grad
    end
    return loss + change
end

mutable struct ChainPV{C,P,G,F,A,Pr,As,Ai}
    β::Float64 # β is a component of the actor.
    chain::C
    params::P
    grads::G
    prefun::F # state -> Vector(Float32) (with β)
    optimizer::A
    x::ElasticArray{Float32} # data buffer to reduce number of allocations
    y::Vector{CombinedDivTarget}
    memory::Int # memory size
    # a special object which encodes the target
    sdims::Int # number of state dimensions (logβ will be appended)
    adims::Int # number of action dimensions (value will be appended)
    prior::Pr
    action_space::As
    action_index::Ai
end

function init_chainpv(ctrl, arch...; memory = 32, sdims = 1, adims = length(ctrl.action_space), β = 1.0, prefun = s -> Float32.(vcat(s...,log(β))))
    chain = SimpleChain(static(sdims+1), arch..., TurboDense(identity, adims+1)) 
    params = SimpleChains.init_params(chain)
    grads = similar(params)
    optimizer = SimpleChains.ADAM()
    ChainPV(β, 
        chain, params, grads, prefun, optimizer, 
        ElasticArray{Float32}(undef, sdims + 1, 0), 
        Vector{CombinedDivTarget}(), 
        memory, sdims, adims, ctrl.action_prior, ctrl.action_space, 
        Dict(Pair.(ctrl.action_space,1:length(ctrl.action_space))) 
        )
end

function (a::ChainPV)(state, action)
    πvec = [a.prior(state,b) for b in a.action_space]
    πvec .*= inv(sum(πvec))
    ii = a.action_index[action]
    wvec = a.chain(a.prefun(state), a.params)
    isnan(wvec[ii]) && error("state looks bad $(print(a.params))")
    isnan(wvec[end]) && error("val looks bad $(print(a.params))")
    return (wvec[ii] - log(sum(πvec[i] * exp(wvec[i]) 
        for i in eachindex(πvec))) + wvec[end]) / a.β
end

function div_target(cpv::ChainPV, qe::QEstimate)
    s = qe.sa0.state
    pivec = [cpv.prior(s,a) for a in cpv.action_space]
    pivec .*= inv(sum(pivec))
    V_t = qe.sa0.V
    eta_t = cpv.β * (qe.sa0.actorq - V_t)
    criticeta = cpv.β * (qe.sa1.criticq - V_t)
    actioni = cpv.action_index[qe.sa0.action]
    logZ_t =qe.logz0
    CombinedDivTarget(pivec, eta_t, criticeta, actioni, V_t, logZ_t, cpv.β)
end

function add_data!(cpv::ChainPV, qe)
    targ = div_target(cpv, qe)
    input = cpv.prefun(qe.sa0.state)
    if length(cpv.y) < cpv.memory
        push!(cpv.y, targ)
        append!(cpv.x, input)
    else # choose a random datapoint to overwrite
        i = rand(1:cpv.memory)
        for j in eachindex(input)
            cpv.x[j,i] = input[j]
        end
        cpv.y[i] = targ
    end
end


function train!(cpv::ChainPV, memory)
    #for qe in memory
    # we only use the final value to to fair training comparisons
    if length(memory) > 0
        qe = memory[end]
        update!(cpv, qe; epochs = 10)
    #end
    #l = length(memory)
    resize!(memory,0) # remove everything, it's been used
        return 1
    else
        return 0
    end
end

function update!(cpv::ChainPV, qe::QEstimate; epochs = 10)
    add_data!(cpv, qe)
    model_loss = SimpleChains.add_loss(cpv.chain, CombinedDivLoss(cpv.y));
    SimpleChains.train_unbatched!(cpv.grads, cpv.params, model_loss, cpv.x, cpv.optimizer, 
        epochs); 
end
