# PIQL

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chelate.github.io/PIQL.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://chelate.github.io/PIQL.jl/dev/)
[![Build Status](https://github.com/chelate/PIQL.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/chelate/PIQL.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/chelate/PIQL.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/chelate/PIQL.jl)


## Main Structures and interfaces


The structure which defines a problem to be solved is called `ControlProblem`

```julia
struct ControlProblem{A, U, P, C, T, W}
    action_space::Vector{A} # something that we can iterate over
    action_prior::U # π(s,a) -> Float64 exactly like energy
    propagator::P # p(x0, a) -> x1 ("random" state)
    cost_function::C # c(x0, a, x1) -> Cost ::Float64
    terminal_condition::T # T(x) -> bool
    initial_state::W # W() -> x0 generates initial states of interest
    γ::Float64 # positive number discount over time, determines path length
end
```
The fundamental unit of policy data is the `StateAction` pair with actor and critic energies

```julia
struct StateAction{S,A} # static and constructed on forward pass
    # atomic unit of data for all reinforcement learning
    state::S
    action::A
    β::Float64 # the beta under which the temperature is allowed to fluctuate
    E_actor::Float64
    E_critic::Float64
    cost::Float64
    f::Float64
    u::Float64
end
```
Via back propagation, this becomes our fundamental unit for training
```julia
struct EnergyEstimate{S,A}
    state::S
    action::A
    β::Float64
    xi::Float64 # energy fluctuation realization
    logz::Float64 # not neccesary except for turning PIQL into Qlearnng
end
```
### Actor interface
Each kind of actor is defined via it's own struct. This struct must be a callable function of state and action, for instance

```julia
function (a::TabularActor)(state,action)
    a.energy[(state,action)]
end
```
 One of the fields must be `β`, the current temperature specifiction. Mid epoch annealing is theoretically possible, but not currently allowed.

 Actors are trained via a dispatched function
```julia
function train!(a::TabularActor, ee::EnergyEstimate)
    ...
end
```




## Folders
### `src` 

We have three folders

`Actors` is where we store representations of state. Tabular actor, deep actor etc.

`Learners` is where we store learning algorithms including different annealing schedules, cost functions etc.

`ControlProblems` is where we store different problems.



`Scripts` 

we have scripting files and notebooks that call using PIQL and then run ploting tesing

# To Do

- Treat paths memory-aggressively.
- prepare for GZL
- get maze 

paper:
- update references
- write up q-learning