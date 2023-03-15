mutable struct TabularPV{S,P,G,A}
    policy::Dict{S,P} # state -> policy mapping
    value::Dict{S,Float64}
    mapping::G # mapping(state) -> . statae key reducing size  of space and enforcing boundaries 
    # mapping(state,action) -> key
    action_space::A # mapping from action to index
    β::Float64
end

function make_tabularpv(ta::TabularActor, ctrl, states; mapping = identity)
    value = Dict(collect(Tuple(state)) => free_energy(collect(Tuple(state)), ctrl, ta) for state in states)
    policy = Dict(collect(Tuple(state)) => tabular_actor_policy(collect(Tuple(state)), ctrl, ta) for state in states)
    action_space = Dict(a => ii for (ii,a) in pairs(ctrl.action))
    TabularPV(policy,value,mapping,action_space,ta.β)
end


function (ta::TabularPV)(state,action)
    key = ta.mapping(state)
    if haskey(ta.value,key)
        v = ta.value[key]
    else
        v = 0
    end
    if haskey(ta.policy,key)
        eta = ta.policy[key].eta[action_space[action]]
    else
        eta = 0
    end
    return - (eta / ta.β + v)
end

function tabular_actor_policy(state, ctrl, ta)
    actions = ctrl.action_space
    prior = normalize([ctrl.prior(state,action) for action in actions])
    free_energy = free_energy(state, ctrl, ta)
    etavec = [ta.β .* (free_energy - ta(state,action)) for action in actions]
    eta = copy(etavec)
    H = copy(etavec)
    wvec = ones(length(etavec))
    Policy(etavec,# predicted, unnormalized
    wvec,
    H,
    eta,
    prior)
end



mutable struct Policy
    etavec::Vector{Float64} # predicted, unnormalized
    wvec::Vector{Float64} 
    H::Vector{Float64} # numerically inferred
    eta::Vector{Float64}
    prior::Vector{Float64} # assumed normalized
end

function normalize(vec)
	vec ./ sum(vec)
end

function backtracking!(H,loss,g,t; β = 0.7)
	while loss(H - g*t*β) < loss(H - g*t)
		 t = t*β
	end
	while loss(H - g*t/β) < loss(H - g*t)
		 t = t/β
	end
	H1 = H - g*t
	return (H1 .- mean(H1), t)
end

function update_policy!(p::Policy)
    (;etavec, prior) = policy
	wvec = normalize(p.wvec)
	function eta(H)
		H .- log(sum(prior .* exp.(H)))
	end
	function loss(H)
		sum(wvec .* (eta(H) .- etavec).^2) / 2
	end
	function l(H)
		wvec .* (eta(H) .- etavec)
	end
	function g(H)
		l(H) .- (sum(l(H)) .* normalize(prior .* exp.(H)))
	end
	t = 1.0
	while true
		(H1,t) = backtracking!(p.H,loss,g(H),t; β = 0.5)
		if sum(abs.(H1 .- p.H)) < 1e-4
			break
		end
		p.H .= H1
	end
	p.eta = eta(p.H)
end


function train(ta::TabularActor, ctrl, states; mapping = identity)
    value = Dict(collect(Tuple(state)) => free_energy(collect(Tuple(state)), ctrl, ta) for state in states)
    policy = Dict(collect(Tuple(state)) => tabular_actor_policy(collect(Tuple(state)), ctrl, ta) for state in states)
    action_space = Dict(a => ii for (ii,a) in pairs(ctrl.action))
    TabularPV(policy,value,mapping,action_space,ta.β)
end

