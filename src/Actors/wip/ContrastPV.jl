export make_contrastpv


# This is the same as TabularPV only we change the type of TabularPolicy to dispatch on a a contrast trainer.
# did not perform impressively


function make_contrastpv(ta, ctrl, states; state_map = identity)
    policy = make_contrastpolicy(ta, ctrl, states; state_map)
    value = make_tabularvalue(ta, ctrl, states; state_map)
    β = ta.β
    PVActor(policy, value, β)
end


# Policy
# returns η_i

# struct PolicyTable
#     wvec::Vector{Float64} # unnormalized weights 
#     prior::Vector{Float64} # assumed normalized
#     visits::Vector{Int64}
# end

mutable struct ContrastPolicy{S,F,G,A}
    ptable::Dict{S,PolicyTable}
    update::F  # (olde, visits, new) -> newavg  updates energies and visits according the the learning rule
    state_map::G # mapping(state) -> S reducing size of space and enforcing boundaries 
    action_index::A # acton_map(A) -> Int index
end


function (tp::ContrastPolicy)(state,action)
    s = tp.state_map(state)
    ii = tp.action_index(action)
    return eta(tp.ptable[s], ii)
end

"""
struct ContrastPair{S,A}
    state::S
    action0::A
    action1::A
    eta0::Float64 # π exp(η)  = probably
    eta1::Float64
    criticeta0::Float64
    criticeta1::Float64
    ftrace::Float64 # Σ_i p_i (1 - p_i)
end
"""

function train!(tp::ContrastPolicy, qe)
    (;state, action0, action1, eta0, eta1, criticeta0, criticeta1, ftrace) = qe.sa0.contrast
    ptab = tp.ptable[state] # policy table
    ptab.visits[1] += 1 
    ii0 = tp.action_index(action0)
    ii1 = tp.action_index(action1)
    importance_weight = ftrace * exp( eta(ptab, ii0) + eta(ptab, ii1) - (eta0 + eta1)) * tp.update(ptab.visits[1]) / 4
    change = (eta(ptab, ii0) -  criticeta0 - (eta(ptab, ii1) - criticeta1)) * importance_weight
    # if  eta0 is too large  this change will be positive and we will need to subtract it off.
    ptab.wvec[ii0] -= change 
    ptab.wvec[ii1] += change
end

# critcQ(s,a) - V(s)  

function make_contrastpolicy(ta::TabularActor, ctrl, states; state_map = identity)
    ptable = Dict(state => policy_table(state, ctrl, ta) for state in states)
    update = learning_rate(;burnin = 1000, p = 2/3)
    actdict = Dict(a => i for (i,a) in enumerate(ctrl.action_space))
    ContrastPolicy(ptable, update, state_map, a->actdict[a])
end


function learning_rate(;burnin = 1, p = 2/3)
    #return the
    function update(visits)
        (burnin / (burnin + visits))^p #
    end
    return update
end
