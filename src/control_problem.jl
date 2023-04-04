
export ControlProblem, StateAction, initial_state_action, new_state_action
using Statistics

"""
ControlProblem is a struct with fields that are 
    the functions which completely define a KL-control problem
"""
struct ControlProblem{A, U, P, R, T, W}
    action_space::Vector{A} # something that we can iterate over
    action_prior::U # π(s,a) -> Float64 exactly like energy
    propagator::P # p(x0, a) -> x1 ("random" state)
    reward_function::R # r(x0, a, x1) -> reward ::Float64
    terminal_condition::T # T(x) -> bool
    initial_state::W # W() -> x0 generates inital states of interest
    γ::Float64 # positive number less than one discount over time
end

struct StateAction{S,A} # static and constructed on forward pass
    # atomic unit of data for all reinforcement learning
    # includes all information and diagnostics available from a single step.
    state::S
    action::A
    β::Float64 # the beta under which the action was chosen
    actorq::Float64
    criticq::Float64
    reward::Float64 # actually incurred cost entering the state
    V::Float64 # free energy of current state (observed value)
    U::Float64 # average energy of current stte
    prior::Float64
end


"""
Start off a trajectory with a new state action pair
"""
function initial_state_action(ctrl::ControlProblem, actor)
#function intial_state_action(ctrl, actor; critic_samples = 1)
    # begin a trajectory from the initial_state distributon
    # return a StateAction object
    state = ctrl.initial_state() # new_state
    return initial_action(state, ctrl, actor)
end


function initial_action(state, ctrl::ControlProblem, actor)
    (action, actorq, v, u, prior) = choose_action(state, ctrl, actor) # new_action
    cost = 0.0
    criticq = 0.0 # there was no prior state-action pair to be used here
    return StateAction(state, action, actor.β, actorq, criticq, cost, v, u, prior)
end

function criticq(state, action, ctrl, actor; critic_samples = 1)
    function qsample(new_state)
        ctrl.reward_function(state, action, new_state) + ctrl.γ*value_function(new_state, ctrl, actor)
    end
    # helper function capturing input variables
    if ctrl.terminal_condition(state)
        # here we catch the terminal state
        return zero(Float64)
    else
        return mean(qsample(ctrl.propagator(state, action)) for ii in 1:critic_samples)
    end
end


function new_state_action(sa::StateAction{S,A}, ctrl::ControlProblem, actor; critic_samples = 1) where {S,A}
    # atomic unit of state evolution
    state = ctrl.propagator(sa.state, sa.action) # new_state
    (action, actorq, V, U, prior) = ifelse(ctrl.terminal_condition(state), 
        (sa.action, 0.0, 0.0, 0.0, sa.prior), 
        choose_action(state, ctrl, actor))
    if isnan(actorq)
        error("the actorq is the first thing that goes bad")
    end
    critq = criticq(sa.state, sa.action, ctrl, actor; critic_samples)
    cost = ctrl.reward_function(sa.state, sa.action, state)
    if isnan(critq)
        error("the criticq is the first thing that goes bad")
    end
    return StateAction{S,A}(state, action, actor.β, actorq, critq, cost, V, U, prior) # Let the compiler know that it is type invariant
end


## change of sign here becuase sampling seams wrong
function choose_action(state, ctrl, actor)
    priors = [ctrl.action_prior(state,a) for a in ctrl.action_space]
    priors = priors ./ sum(priors)
    Q = [actor(state,a) for a in ctrl.action_space]
    Qmax = maximum(Q)
    Q .-= Qmax
    z = sum(priors .* exp.(actor.β .* Q))
    V = log(z) / actor.β 
    U = sum(priors .* exp.(actor.β .* Q) .* Q) / z
    ii = sample( weights(priors .* exp.(actor.β .* Q)))
    prior = priors[ii]
    return (ctrl.action_space[ii], Q[ii] + Qmax, V + Qmax, U + Qmax, prior)
end



function value_function(state, ctrl, actor)
    priors = [ctrl.action_prior(state, a) for a in ctrl.action_space]
    prior_normalization = sum(priors)
    Q = [actor(state,a) for a in ctrl.action_space]
    Qmax = maximum(Q)
    z = sum(priors.* exp.(actor.β .*(Q .- Qmax))) / prior_normalization
    if isnan(z)
        println(Q)
    end

    #return - log(z) / actor.β
    return log(z) / actor.β + Qmax
end
