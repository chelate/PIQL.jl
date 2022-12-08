
export ControlProblem, StateAction, intial_state_action, new_state_action
using Statistics

"""
ControlProblem is a struct with fields that are 
    the functions which completely define a KL-control problem
"""
struct ControlProblem{A, U, P, C, T, W}
    action_space::Vector{A} # something that we can iterate over
    action_prior::U # π(s,a) -> Float64 exactly like energy
    propagator::P # p(x0, a) -> x1 ("random" state)
    cost_function::C # c(x0, a, x1) -> Cost ::Float64
    terminal_condition::T # T(x) -> bool
    initial_state::W # W() -> x0 generates inital states of interest
    γ::Float64 # positive number discount over time
end

struct StateAction{S,A} # static and constructed on forward pass
    # atomic unit of data for all reinforcement learning
    state::S
    action::A
    β::Float64 # the beta under which the temperature is allowed to fluxuate
    E_actor::Float64
    E_critic::Float64
end


"""
Start off a trajectory with a new state action pair
"""
function intial_state_action(ctrl::ControlProblem, actor; critic_samples = 1)
#function intial_state_action(ctrl, actor; critic_samples = 1)
    # begin a trajectory from the initial_state distributon
    # return a StateAction object
    state = ctrl.initial_state() # new_state
    (action, E_actor) = choose_action(state, ctrl, actor) # new_action
    E_critic = 0.0 # there was no prior state-action pair to be used here
    return StateAction(state, action, actor.β, E_actor, E_critic)
end



function new_state_action(sa::StateAction{S,A}, ctrl::ControlProblem, actor; critic_samples = 1) where {S,A}
    # atomic unit of state evolution
    state = ctrl.propagator(sa.state, sa.action) # new_state
    if ctrl.terminal_condition(state)
        (action, E_actor) = (sa.action, 0.0) # place holder 
        else
        (action, E_actor) = choose_action(state, ctrl, actor) # new_action
    end
    if isnan(E_actor)
        println("the E_actor is the first thing that goes bad")
        #break
    end

    E_critic = energy_critic(sa.state, sa.action, ctrl, actor; critic_samples)
    if isnan(E_critic)
        println("the E_critic is the first thing that goes bad")
        #break
    end
    return StateAction{S,A}(state, action, actor.β, E_actor, E_critic) # Let the compiler know that it is type invariant
end


## change of sign here becuase sampling seams wrong
function choose_action(state, ctrl, actor)
    priors = [ctrl.action_prior(state,a) for a in ctrl.action_space]
    priors = priors ./ sum(priors)
    energies = [actor(state,a) for a in ctrl.action_space]
    emin=minimum(energies)
    ii = sample(weights(priors .* exp.(- actor.β .* (energies .- emin))))
    return (ctrl.action_space[ii], energies[ii])
end

function energy_critic(state, action, ctrl, actor; critic_samples = 1)
    function energy_sample(new_state)
        ctrl.cost_function(state, action, new_state) + ctrl.γ*free_energy(new_state, ctrl, actor)
    end
    # helper function capturing input variables

    if ctrl.terminal_condition(state)
        # here we catch the terminal state
        return zero(Float64)
    else
        return mean(energy_sample(ctrl.propagator(state, action)) for ii in 1:critic_samples)
    end
end

## again sign was wrong I think
function free_energy(state, ctrl, actor)
    priors=[ctrl.action_prior(state, a) for a in ctrl.action_space]
    prior_normalization=sum(priors)
    #prior_normalization = sum( ctrl.action_prior(state, a) for a in ctrl.action_space)
    energies=[actor(state,a) for a in ctrl.action_space]
    emin=minimum(energies)
    # z = sum( ctrl.action_prior(state, a) * exp(-actor.β * actor(state,a))
    #    for a in ctrl.action_space) / prior_normalization
    z = sum(priors.* exp.(-actor.β .*(energies .-emin)))/prior_normalization
    #z = sum( ctrl.action_prior(state, a) * exp(-actor.β * actor(state,a))
    #    for a in ctrl.action_space) / prior_normalization
    if isnan(z)
        println(energies)
        println(emin)
        println(exp.(-actor.β .*(energies .-emin)))
    end

    #return - log(z) / actor.β
    return - log(z)/ actor.β + emin
end
