export binary_bandit_problem

using Distributions

# the state structure is 
# state = (hi,ti),...,rounds_left)

struct BanditState{I<:AbstractVector,R<:Integer}
    belief::I
    rounds::R
end

function observe(bs,a)
    (h,t) = bs.belief[a]
    rand(Bernoulli(h/(h+t)))
end

function update(bs,a,o)
    belief = copy(bs.belief)
    (h,t) = belief[a]
    belief[a] = (h + o, t + !o)
    BanditState(belief, bs.rounds -1)
end

function reachable(bs)
    [update(bs,a,o) for a in 1:length(bs.belief) for o in [true,false]]
end


function binary_bandit_problem(;rounds = 10, ncoins = 2, prior = (1,1), payoff = 1.0, γ = 1.0)
    action_space = 1:ncoins
    action_prior(s,a) = 1.0 / ncoins
    function propagator(s0,a)
        o = observe(s0, a)
        return update(s0, a, o)
    end
    function reward(s0, a, s1)
        (s1.belief[a][1] - s0.belief[a][1])*payoff
    end
    function propagator_average(s,a,f)
        (h,t) = s.belief[a]
        f(update(s, a, false))*(t/(h+t)) + f(update(s, a, true))*(h/(h+t))
    end
    term(bs) = bs.rounds < 1
    initial_state() = BanditState([prior for ii in 1:ncoins],rounds)
    ControlProblem(
        action_space,    
        action_prior,
        propagator,
        reward,
        propagator_average,
        term, initial_state, γ)
end


function state_iterator(;rounds = 10, ncoins = 2, prior = (1,1))
    out = [BanditState([prior for ii in 1:ncoins],rounds)]
    cur = out
    for _ in 1:rounds-1
        next = similar(cur,0)
        for state in cur
            push!(next, reachable(state)...)
        end
        vcat(out,next)
        cur = next
    end
    return Iterators.reverse(out)
end