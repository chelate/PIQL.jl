using Revise
using PIQL
using StatsBase


gw = make_gridworld([20,20]; density = 0.30);
ctrl = make_ctrl(gw; γ = .9999);
states = state_iterator(gw)
actor0 = get_ideal_actor(ctrl, states)
actor_jittered = jitter_actor(actor0, ctrl, states; jitter =  0.5)
actor_heated = get_ideal_actor(ctrl, states; β = 2.0)
actor_heated.β = 1.0

actor_cooled = get_ideal_actor(ctrl, states; β = 0.5)
actor_cooled.β = 1.0

function getstart(gw, actor1, ctrl)
    # finds a nice start location fairly far from the goal
    start = PIQL.initial_state_action(ctrl,actor1)
    while sum(abs.(start.state .- gw.goal)) < 10
        start = PIQL.initial_state_action(ctrl,actor1)
    end
    return start
end
start = getstart(gw, actor0, ctrl)

function energy_est_truth(ctrl, actor0, actor1; 
    depth = 1, 
    sa = PIQL.initial_state_action(ctrl,actor1))
    piql = PIQL.random_piql(ctrl, actor1; depth, sa)
    while isempty(piql.memory)
        piql = PIQL.random_piql(ctrl, actor1; depth)
    end
    ee = last(piql.memory)
    current_e = actor1(ee.state, ee.action)
    (actor0(ee.state, ee.action) , ee.xi, current_e)
    # truth, est, current
end



expvec(ctrl, actor0, actor1; sa = start) = [mean(
    (x-> exp(-actor0.β * (x[2] -  x[1])))(energy_est_truth(ctrl, actor0, actor1; depth = d, sa)) for ii in 1:1000) for d = 1:50]

expvec(ctrl, actor0, actor_jittered)

lineplot(log.(expvec(ctrl, actor0, actor_jittered)))
lineplot(log.(expvec(ctrl, actor0, actor_cooled)))


"""
For plotting the result, show the free energy as a function of state
This might be helpful.
"""
function free_energy_matrix(ctrl, actor, gw; max = 60.0)
    out = fill(NaN,size(gw.walls))
    states = state_iterator(gw)
    for ii in states
        state = collect(Tuple(ii))
        out[ii] = PIQL.free_energy(state,ctrl,actor)
    end
    out[out .> max] .= max
    out[isnan.(out)] .= max
    return out
end

free = free_energy_matrix(ctrl, actor0, gw)
free1 = free_energy_matrix(ctrl, actor1, gw)

run_piql!(piql, ctrl, actor)

training_epoch!(piql, ctrl, actor)


for ii in 1:100
    training_epoch!(piql, ctrl, actor)
end


"""
We test the accuracy of this ideal actor against the theory i.e. is the energy the cost to go for a state action pair, and is the free energy the cost for a state
"""

# priors = [ctrl.action_prior(start.state,a) for a in ctrl.action_space]
# priors .= priors ./ sum(priors)
# energies = [actor0(start.state,a) for a in ctrl.action_space]
# min = minimum(energies)
# energies .-= min
# z = sum(priors .* exp.(-actor0.β .* energies))
# f = - log(z) / actor0.β
# u = sum(priors .* exp.(-actor0.β .* energies) .* energies) / z


start_cost_mc = mean((x->(x[1] + x[2]))(performance_estimate(ctrl, actor0; state = start.state)) for ii in 1:10000)
cost_std = sqrt(var((x->(x[1] + x[2]))(performance_estimate(ctrl, actor0; state = start.state)) for ii in 1:10000))/sqrt(10000)
start_cost_free = - log(sum(ctrl.action_prior(start.state,a) * exp(-actor0.β * actor0(start.state, a)) 
    for a in ctrl.action_space) / sum(ctrl.action_prior(start.state,a) for a in ctrl.action_space)) / actor0.β    

4 > abs(start_cost_mc - start_cost_free) / cost_std # should be of order 1, true

"""
Here we test whether a piql terminating on the true action gives the unbiased estimate of the true energy
struct StateAction{S,A} # static and constructed on forward pass
    state::S
    action::A
    β::Float64 # the beta under which the temperature is allowed to fluxuate
    E_actor::Float64
    E_critic::Float64
    cost::Float64 # actually incurred cost
    f::Float64 # free energy of current action
    u::Float64 # average energy of current action
end
"""

actor_jittered = jitter_actor(actor0, 5.0);

function energy_estimate(start, actor0, actor1, ctrl; depth = 1)
    # backpropagates with the actor0 free energy at the end.
    # should be unbiased at any depth.
    piql = PIQL.random_piql(ctrl, actor_jittered; depth = 1, sa = start)
    lst = piql.worldline[end]
    jfn(actor) = - log(sum(
        ctrl.action_prior(lst.state,a) * exp(-actor.β * actor(lst.state, a)) 
            for a in ctrl.action_space) / sum(ctrl.action_prior(lst.state,a) for a in ctrl.action_space)
    ) / actor.β
    zlst = exp(actor1.β * (jfn(actor1) - jfn(actor0))) 
    energy_est = lst.E_critic - log(zlst(actor0,actor_jittered))/actor_jittered.β
    for ii in 1:depth-1
        sa = piql.worldline[ii]
        energy_est += sa.β * (sa.E_actor[ii+1] - sa.E_critic[ii])
    end
    return energy_est
end


piql = PIQL.random_piql(ctrl, actor_jittered; depth = 1, sa = start)


eest = lst.E_critic - log(zlst(actor0,actor_jittered))/actor_jittered.β
actor_jittered(start.state, start.action)
real = actor0(start.state, start.action)



lastj1 = - log(sum(
    ctrl.action_prior(lst.state,a) * exp(-actor_jittered.β* actor0(lst.state, a)) 
        for a in ctrl.action_space) / sum(ctrl.action_prior(lst.state,a) for a in ctrl.action_space)) / actor0.β





using UnicodePlots

a = zeros(size(gw.walls))
a[gw.goal...] += 2
a .+= gw.walls
heatmap(a)
heatmap(free)
heatmap(free1)