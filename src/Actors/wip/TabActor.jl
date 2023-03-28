export TabActor, tab_actor, train!



# exact repurposing of previous but with a different training algorithm
# which attempts to update Q and V simultaneously

"""
major difference: update rule is now mediated by a leraning rate function directly
on both the free energy and the Q functions.

Note: this does NOT seem to work. :(
"""

mutable struct TabActor{SA,F,G,A}
    qtable::Dict{SA,Float64}
    visits::Dict{SA,Int}
    learning_α::F  # (visit) -> learningrate  updates energies and visits according the the learning rule
    mapping::G # mapping(state, action) -> . key reducing size  of space and enforcing boundaries 
    action_space::A # for scanning for fall back.
    # mapping(state,action) -> key
    αpower::Float64
    β::Float64
end

function learning_rate(;burnin = 1, p = 2/3)
    function update(visits)
        (burnin / (burnin + visits))^p
    end 
    return update
end

function tab_actor(ta::TabularActor; learning_α = learning_rate(;burnin = 1, p = 2/3), αpower = 1/2)
    (;qtable, visits, mapping, action_space, β) = ta
    TabActor(qtable, visits, learning_α, mapping, action_space, αpower, β)
end

function (ta::TabActor)(state,action)
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
function train!(ta::TabActor, memory)
    # for ee in memory
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
        actionvisits = ta.visits[key]
        statevisits = 0 
        for a in ta.action_space
            if haskey(ta.visits, ta.mapping(qe.sa0.state, a))
                statevisits += ta.visits[ta.mapping(qe.sa0.state, a)]
            end
        end
        αQ = ta.learning_α(actionvisits)
        pa = qe.sa0.prior * exp(qe.sa0.β * (qe.sa0.actorq - qe.sa0.V))
        pabar = -expm1(qe.sa0.β * (qe.sa0.actorq-qe.sa0.V) + log(qe.sa0.prior))
        # pa = actionvistis/statevisits
        α_ratio = pa^ta.αpower
        r = (α_ratio - pa) / (1.0 - pa)        # if α_ratio <= pa
        #     r = 0.0
        # else
        #     r = ((α_ratio - pa) / pabar)
        # end
        # if !(0 <= r <= 1)
        #     print("r = $r ")
        #     print("α_ratio = $α_ratio")
        #     print("pa = $pa ")
        #     error("αV = $αV ")
        # end
        # # for numerical stability
        ΔQ = log1p(αQ * expm1(qe.logz0)) / qe.sa0.β
        Δc = log1p(r * αQ * expm1(qe.logz0)) / qe.sa0.β
        if !isfinite(ΔQ)
            print("αQ = $αQ ")
            print("loz0 =  $(qe.logz0) ")
            error("ΔQ = $ΔQ")
        elseif !isfinite(Δc)
            error("Δc = $Δc")
        end
        # update energy
        if haskey(ta.qtable,key)
            ta.qtable[key] += ΔQ
        else
            push!(ta.qtable, key => ΔQ)
        end
        for a in ta.action_space
            if a != qe.sa0.action
                k = ta.mapping(qe.sa0.state, qe.sa0.action)
                if haskey(ta.qtable, k)
                    ta.qtable[k] += Δc
                else
                    push!(ta.qtable, key => Δc)
                end
            end
        end
    #end
    resize!(memory,0) # remove everything, it's been used
    end
end


##
#
# need to decide what the q learning rule is.
#
#