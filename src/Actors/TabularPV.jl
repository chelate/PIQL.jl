export make_tabularpv

"""
We write this as a separation between policy and value function
The policy is representable as a set of unnormalized weights
w_i so that 

log(p_i) = log(w_i) - log(sum(w))
 η_i = log(w_i/π_i) - log(sum(w))
 η_i = ω_i - log(sum(π_i * exp(ω_i)))

 The ω_i = log(w_i/π_i) have an estimator in

β(criticq - actorv)_i = hatω_i => ω_i

There's no exponential sampling that needs to be averaged so we can simply update (with a pretty strong learning rate.)

ω_i(t+1) = ω_i(t) + α (hatω_i -  ω_i(t))

at the same time, the actorq can be generated

actorq_i = η_i / β + actorv_i = (ω_i - log(Ξ)) / β + actorv

and it can be constructed from 

actorq_i = η_i / β + actorv_i = (q_i - log(Ξ)) / β + actorv

v = sum(π_i exp(β*actor_qi)) / β
w = β * (actorq_i - v)

Still experimental, loss interest after TabActor update rule didn't work.
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
    for qe in memory
    # we only use the final value to to fair training comparisons
    #if length(memory) > 0
        #qe = memory[end]
        train!(pv.value, qe)
        train!(pv.policy, qe)
    end
    l = length(memory)
    resize!(memory,0) # remove everything, it's been used
    return l
    #end
end
# initialize
function make_tabularpv(ta, ctrl, states; state_map = identity)
    policy = make_tabularpolicy(ta, ctrl, states; state_map)
    value = make_tabularvalue(ta, ctrl, states; state_map)
    β = ta.β
    PVActor(policy, value, β)
end


# Value

mutable struct TabularValue{S,F,G}
    vtable::Dict{S,Float64}
    visits::Dict{S,Int}
    update::F  # (olde, visits, new) -> newavg  updates energies and visits according the the learning rule
    state_map::G # mapping(state) -> S reducing size  of space and enforcing boundaries 
    # mapping(state,action) -> key
end
# access
function (tp::TabularValue)(state)
    s = tp.state_map(state)
    tp.vtable[s]
end
# training
function train!(tv::TabularValue, qe)
    key = tv.state_map(qe.sa0.state)
    tv.visits[key] += 1
    tv.vtable[key] += tv.update(tv.vtable[key], tv.visits[key], vbound(qe))
end
# Initialization
function make_tabularvalue(ta::TabularActor, ctrl, states; state_map = identity)
    vtable = Dict(state => value_function(state, ctrl, ta) for state in states)
    visits = Dict(state => 1 for state in states)
    update = update_function_piql(;burnin = 1, p = 2/3, β = ta.β)
    TabularValue(vtable, visits, update, state_map)
end

# Policy
# returns η_i

struct PolicyTable
    wvec::Vector{Float64} # log w_i/p_i
    prior::Vector{Float64} # assumed normalized
    visits::Vector{Int64}
end

mutable struct TabularPolicy{S,F,G,A}
    ptable::Dict{S,PolicyTable}
    update::F  # (olde, visits, new) -> newavg  updates energies and visits according the the learning rule
    state_map::G # mapping(state) -> S reducing size of space and enforcing boundaries 
    action_index::A # acton_map(A) -> Int index
end

eta(p::PolicyTable, ii) = p.wvec[ii] - log(sum(p.prior .* exp.(p.wvec)))
normalize(vec) = vec ./ sum(vec)


function (tp::TabularPolicy)(state,action)
    s = tp.state_map(state)
    ii = tp.action_index(action)
    return eta(tp.ptable[s], ii)
end

function train!(tp::TabularPolicy, qe)
    s = tp.state_map(qe.sa0.state)
    ii = tp.action_index(qe.sa0.action)
    ptab = tp.ptable[s]
    ptab.visits[ii] += 1
    ptab.wvec[ii] += tp.update(ptab.wvec[ii], ptab.visits[ii], ηbound(qe))
end

# critcQ(s,a) - V(s)  

function make_tabularpolicy(ta::TabularActor, ctrl, states; state_map = identity)
    ptable = Dict(state => policy_table(state, ctrl, ta) for state in states)
    update = update_policy_table(;burnin = 1, p = 2/3, β = ta.β)
    actdict = Dict(a => i for (i,a) in enumerate(ctrl.action_space))
    TabularPolicy(ptable, update, state_map, a->actdict[a])
end

function policy_table(state, ctrl, ta)
    # for initialization
    prior = normalize([ctrl.action_prior(state,a) for a in ctrl.action_space])
    wvec = [ta.β * ta(state,a) for a in ctrl.action_space]
    wvec = wvec .- log(sum(prior .* exp.(wvec)))
    visits = ones(Int64, length(wvec))
    PolicyTable(wvec, prior, visits)
end

function update_policy_table(;burnin = 1, p = 2/3, β = ta.β)
    #return the
    function update(olde, visits, new)
        γ = (burnin / (burnin + visits))^p
        return log1p(γ * expm1(β * (new - olde))) / β
        # using the piql update rule, linear in the exponent
    end 
    return update
end






function tabular_actor_policy(state, ctrl, ta)
    actions = ctrl.action_space
    prior = normalize([ctrl.action_prior(state,action) for action in actions])
    actorv = value_funciton(state, ctrl, ta)
    w = [ta.β .* (ta(state,action) - actorv) for action in actions]
    Policy(w,prior)
end


# function backtracking!(H,loss,g,t; β = 0.7)
# 	while loss(H - g*t*β) < loss(H - g*t)
# 		 t = t*β
# 	end
# 	while loss(H - g*t/β) < loss(H - g*t)
# 		 t = t/β
# 	end
# 	H1 = H - g*t
# 	return (H1 .- mean(H1), t)
# end

# function update_policy!(p::Policy)
#     (;etavec, prior) = policy
# 	wvec = normalize(p.wvec)
# 	function eta(H)
# 		H .- log(sum(prior .* exp.(H)))
# 	end
# 	function loss(H)
# 		sum(wvec .* (eta(H) .- etavec).^2) / 2
# 	end
# 	function l(H)
# 		wvec .* (eta(H) .- etavec)
# 	end
# 	function g(H)
# 		l(H) .- (sum(l(H)) .* normalize(prior .* exp.(H)))
# 	end
# 	t = 1.0
# 	while true
# 		(H1,t) = backtracking!(p.H,loss,g(H),t; β = 0.5)
# 		if sum(abs.(H1 .- p.H)) < 1e-4
# 			break
# 		end
# 		p.H .= H1
# 	end
# 	p.eta = eta(p.H)
# end


# function train!(tp::TabularPV, memory)
#     # for ee in memory
#     # we only use the final value to to fair training comparisons
#     if length(memory) > 0
#     qe = memory[end]
#         key = ta.mapping(qe.sa0.state)
#         #  update visits
#         if haskey(ta.visits, key)
#             ta.visits[key] += 1
#         else
#             push!(ta.visits, key => 1)
#         end
#         # update energy
#         if haskey(ta.qtable,key)
#             ta.qtable[key] += ta.update(ta.qtable[key], ta.visits[key], qbound(qe))
#         else
#             push!(ta.qtable, key => qbound(qe))
#         end
#     #end
#     resize!(memory,0) # remove everything, it's been used
#     end
# end
