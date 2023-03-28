using IntervalSets

# state of cart state = (x, θ, p_x, p_θ)

## action_space::Vector{A} # something that we can iterate over
## action_prior::U # π(s,a) -> Float64 exactly like energy
## propagator::P # p(x0, a) -> x1 ("random" state)
## cost_function::C # c(x0, a, x1) -> Cost ::Float64
## terminal_condition::T # T(x) -> bool
# initial_state::W # W() -> x0 generates inital states of interest
# β::Float64 # positive number parameterizing kl-cost of prior deviation
# γ::Float64 # positive number discount over time

# θ is zero at the top (I know this is a dumb choice)

function pendulum_control_problem(; g = 9.8, m1 = 1.0, L = 1.0, 
    δt = .01, control_force = 1.0, damping = 0.1, potential_sharpness = 1)
    action_space = [[-1.],[0.],[1.]] .* control_force .* sqrt(δt)
    action_prior = (s,a) -> 1
    ControlProblem(
        action_space, action_prior,
        (state,action) -> pendulum_euler_step(state, action; g, m1, L, δt, damping),
        (s0,a,s1) -> pendulum_cost(s0; potential_sharpness), 
        state->pendulum_stop(state; end_time = 50*sqrt(L/g)), pendulum_start
    )
end

function pendulum_cost(state0; potential_sharpness = 1)
    θ = state0[2]
    return (1 - cos(θ))^(1/potential_sharpness)
end

function pendulum_start()
    [π, 0.0, 0.0]
end

function pendulum_stop(state; θ_tol = 2*10^-2, pθ_tol = 2*10^-2, px_tol = 2*10^-2, end_time = 2*10^2)
    (θ, pθ, t) = state
    # return reduce(&, [abs(mod(θ - 0, 2*pi)) < θ_tol, abs(pθ - 0) < pθ_tol, abs(px - 0) < px_tol]) |
    return (t > end_time)
end

# action space = [-1,0,1]

function pendulum_euler_step(state, action; g = 9.8, m1 = 1.0, L = 1.0, δt =0.01, damping = 0.1) 
    (θ, pθ, t) = state
    # pθ^2 /(2ml^2) + mgl(1-cos(θ))
    θ_dot = pθ / (m1*L^2)
    pθ_dot = g*L*m1*sin(θ) # h_θ
    newstate = [
        mod(θ + θ_dot * δt,-π..π), 
        pθ + pθ_dot * δt - θ_dot * damping * δt + action[1],
        t + δt] #also really slow
    return newstate
end



# function cart_hamiltonian(p, q, control; g = 9.8, m1 = 2.0 , m2 = 1.0, L = 0.5) 
#     (x, θ) = q
#     (p_x, p_θ) = p
#     h =  g * L * m2 * cos(θ) + 
#         (L^2 * m2 * p_x^2 + (m1 + m2)* p_θ^2 - 
#         2 *L * m2 * p_x * p_θ * cos(θ) ) / 
#         ( 2 * L^2*m2*(m1 + m2 * sin(θ)^2)) 
#         - control * x # this is the control, an added linear potential that kicks the system
#     return h
# end

# function initialize(p,q,c)
#     a = HamiltonianProblem(cart_hamiltonian,p,q,[0.0,1.0],c)
#     return init(a, Tsit5())
# end

# function update(s,c; dt = .2)
#    (t,p,q) = s
#    integrator = initialize(p,q,c)
#    step!(integrator, dt, true)
#    (p1,q1) = integrator.u
#    return (t+dt,p1,q1)
# end

# @time update((1.0,[0.1,0.2],[.3,.4]),.2)