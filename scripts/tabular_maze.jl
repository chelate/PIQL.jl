
begin
using Revise
using PIQL
using StatsBase
end

begin 
gw = make_gridworld([20,20]; density = 0.30);
ctrl = make_ctrl(gw; γ = .9999);
states = state_iterator(gw);
actor0 = get_ideal_actor(ctrl, states);
actor_jittered = jitter_actor(actor0, ctrl, states; jitter =  0.5);
actor_heated = get_ideal_actor(ctrl, states; β = 2.0);
actor_heated.β = 1.0;
actor_cooled = get_ideal_actor(ctrl, states; β = 0.5);
actor_cooled.β = 1.0;
vectorstates = collect.(Tuple.(states))
end

excess_entropy(ctrl, actor0, actor_cooled, vectorstates)


pv_jittered = make_tabularpv(actor_jittered, ctrl, vectorstates);
pv_heated = make_tabularpv(actor_heated, ctrl, vectorstates);
pv0 = make_tabularpv(actor0, ctrl, vectorstates);

 pv_heated = make_contrastpv(actor_jittered, ctrl, vectorstates);


"""
Below was an attempt to train free energy and Q function simultaneously.
    
    Unfortunately it turned out to be numerically unstable leading to enormously high logZ and nan-ing 
"""
# ta_jittered = tab_actor(actor_jittered)
# ta_heated = tab_actor(actor_heated)
# ta_cooled = tab_actor(actor_cooled)

"""
plans for piql meeting
    - make training curves for heated cooled jittered
    - make curves for PIQL and Q-learning
    - start pendulum problem.
"""

### test that the true cost is really rcovered by the free energy
# 
# cost_dict = get_cost(ctrl, actor0, states)
# free_dict = get_free_energy(ctrl, actor0, states)
# for ii in states
#     state = collect(Tuple(ii))
#     if abs(cost_dict[state] - free_dict[state]) > 1e-10
#         print("$state, $(cost_dict[state]), $(free_dict[state])")
#     end
# end
# accurate almonst to machine ϵ

c = excess_reward(ctrl, actor0, pv_jittered, states)

function getstart(gw, actor1, ctrl)
    # finds a nice start location fairly far from the goal
    start = PIQL.initial_state_action(ctrl,actor1)
    while sum(abs.(start.state .- gw.goal)) < 10
        start = PIQL.initial_state_action(ctrl,actor1)
    end
    return start
end

start = getstart(gw, actor0, ctrl)

function training_curve(actor1, ctrl, actor0, states; epochs = 10^4)
    actor = deepcopy(actor1)
    piql = PIQL.random_piql(ctrl, actor; depth = 50, sa = start)
    out = Float64[]
    for ii in 1:epochs
        training_epoch!(piql, ctrl, actor)
        if  1 == mod(ii,1000) 
            c = excess_reward(ctrl, actor0, actor, states)
            push!(out,c)
        end
    end
    return out
end

piql = PIQL.random_piql(ctrl, actor0; depth = 30);

for ii in 1:1000
    training_epoch!(piql, ctrl, pv0)
end

c = excess_reward(ctrl, actor0, pv0, states)

training_result = training_curve(actor_heated, ctrl, actor0, states)

using UnicodePlots
lineplot(training_result)



function energy_est_truth(ctrl, actor0, actor1; 
    depth = 1, 
    sa = PIQL.initial_state_action(ctrl,actor1))
    piql = PIQL.random_piql(ctrl, actor1; depth, sa)
    while isempty(piql.memory)
        piql = PIQL.random_piql(ctrl, actor1; depth)
    end
    ee = last(piql.memory)
    current_e = actor1(ee.sa0.state, ee.sa0.action)
    (actor0(ee.sa0.state, ee.sa0.action) , PIQL.qbound(ee), current_e)
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
function free_energy_matrix(ctrl, actor, gw; min = -40.0)
    out = fill(NaN,size(gw.walls))
    states = state_iterator(gw)
    for ii in states
        state = collect(Tuple(ii))
        out[ii] = PIQL.value_function(state,ctrl,actor)
    end
    out[out .< min] .= min
    out[isnan.(out)] .= min .* 1.1
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


free = free_energy_matrix(ctrl, actor0, gw)
free1 = free_energy_matrix(ctrl, pv0, gw)