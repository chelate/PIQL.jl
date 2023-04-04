
using Flux
using Zygote


struct Deep_actor{N,U,O,L,t}
    model::N
    update::U  ## updates energies and visits according the the learning rule
    opt::O
    lossfunction::L
    trainparameters::t
end


function (a::Deep_actor)(state,action)
    sa=append!(deepcopy(state),action)
    return a.model(sa)[1]
end

function initilize_Deep_actor( model_parameters; lossfunction=Flux.Losses.mse,make_model=make_model_3H, lr=0.001)
    function update(actor,gs)
        #ps= Flux.params(actor.model)
        ps=actor.trainparameters
        Flux.update!(actor.opt, ps, gs)
    end
    model_parameters
    model1=make_model(model_parameters)
    opt=ADAM(lr)
    return Deep_actor(model1,update,opt,lossfunction,Flux.params(model1))
end




function make_model_3H(model_parameters::Tuple{Int64, Int64, Int64, Any, Any})
    in_dim, out_dim, base_size, means, stds=model_parameters
    #println(means)
    #println(stds)
    model=Chain(x -> x .- means,
    x -> x ./stds,
    Dense(in_dim => base_size,relu),
    #BatchNorm(base_size),
    Dense(base_size => base_size*2,relu),
    #BatchNorm(base_size*2),
    Dense(base_size*2 => base_size,relu),
    Dense(base_size => out_dim),
    )
    return model
end




#function piql_loss(x::Float64,y::Vector{})
function piql_loss(x,y)
    #println(y)
    log_m = @view y[1,:]
    log_w = @view y[2,:]
    log_z = logsumexp.(log_m,log_w)
    E_A = @view y[3,:]
    β = @view y[4,:]
    #println(β)
    #println(x)
    exponent=exp.(β.*(x.-E_A))
    loss=(exp.(log_m .- log_z) .* log.(1 .+ exponent ) .+ exp.(log_m .- log_z) .* log.(1 .+ 1 ./exponent ) ) ./ β.^2
    return loss
end

function gradient_piql_loss(x,y)
    log_m = @view y[1,:]
    log_w = @view y[2,:]
    log_z = logsumexp.(log_m,log_w)
    E_A = @view y[3,:]
    β = @view y[4,:]
    g= ( exp.(log_m .- log_z) .- (1 ./ (1 .+exp.(β.*(x .- E_A))))) ./ β
    return g
end
#Zygote.@adjoint piql_loss(x, y) = piql_loss(x, y), Δ -> (gradient_piql_loss(x,y), 1,1 , 1,1)
