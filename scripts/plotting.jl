##
using CairoMakie
using ColorSchemes
using Colors
using JLD2
using LaTeXStrings
using Makie.GeometryBasics

function plot_experiment(exp_dict, furthest, trueval; noise = .1)
    f = Figure()
    ax = Axis(f[1,1], xlabel = L"iteration $i$: $V^{i+1} \rightarrow V^{i} + β^{-1}\log Z^{β}$", ylabel = L"residual: $V^i(s_0) - \nu(s_0)$", title = "noise = $noise")
    trs(x) = sign(x)*abs(x)^(1/2)
    prs = sort(collect(pairs(exp_dict)),by = x->-x[1])
    l = length(prs) 
    i = 0
    for (β,d) in prs
        out = map(x -> trs( x(furthest)- trueval(furthest)), d)[2:end]
        lines!(ax,out; label = "β = $(round(β, digits = 3))",
        color = ColorSchemes.roma[i/(length(prs)-1)],
        linewidth = 2.0
        )
        i += 1
    end
    axislegend(ax)
    f
end

function plot_optimality_gap(gap_dict, furthest, trueval; noise = .1)
    f = Figure()
    ax = Axis(f[1,1], xlabel = L"iteration $i$: $V^{i+1} \rightarrow V^{i} + β^{-1}\log Z^{β}$", ylabel = L"optimality gap: $R^i(s_0) - \nu(s_0)$", title = "noise = $noise", yscale = log10)
    prs = sort(collect(pairs(gap_dict)),by = x->-x[1])
    l = length(prs) 
    i = 0
    for (β,d) in prs
        out = map(x ->  x(furthest), d)[2:end]
        lines!(ax,out; label = "β = $(round(β, digits = 3))",
        color = ColorSchemes.roma[i/(length(prs)-1)],
        linewidth = 2.0
        )
        i += 1
    end
    axislegend(ax)
    f
end

function matrixify(g; filling = 0.0)
    s = maximum(k for k in keys(g.dict))
    m = fill(filling,Tuple(s) .+ 1)
    for k in keys(g.dict)
        m[k] = g(k)
    end
    return m
end

function matplot(x) 
    m = matrixify(x; filling = NaN)
    f = Figure()
    ax = Axis(f[1,1], xlabel = L"x", ylabel = L"y", aspect = DataAspect())
    ci = Tuple(argmin(matrixify(x; filling = Inf)))
    cj = Tuple(argmax(matrixify(x; filling = -Inf)))
    hm = heatmap!(ax,m, colormap = :roma, nan_color = RGB(.0))
    poly!(Circle(Point2f(ci...), .2), color = RGB(.3,1.0,.3))
    poly!(Circle(Point2f(cj...), .2), color = RGB(1.0,0.3,.3))
    Colorbar(f[1, 2],hm, label = L"ν(s)")
    f
end

# apply these plots to the saved data

for file in readdir("data")
    out = JLD2.load(joinpath("data",file))["out"]
    name = split(file,".")[1]
    fig = plot_experiment(out.traj, out.s0, out.ν; noise = out.randomness)
    CairoMakie.save(joinpath("figures","experiment_$(name).pdf"),fig)
end

for file in readdir("data")
    out = JLD2.load(joinpath("data",file))["out"]
    name = split(file,".")[1]
    fig = plot_optimality_gap(out.gap, out.s0, out.ν; noise = out.randomness)
    CairoMakie.save(joinpath("figures","gap_$(name).pdf"),fig)
end

begin 
    file = readdir("data")[1]
    out = JLD2.load(joinpath("data",file))["out"]
    fig = matplot(out.ν)
    CairoMakie.save(joinpath("figures","gridworld.pdf"),fig)
end








let out = out
    plot_optimality_gap(out.gap, out.s0, out.ν; noise = out.randomness)
end

out = let gwctrl = gridworld_ctrl(gw; randomness = 0.1, 
        γ = 0.99)
    weaker_gwctrl = modify(gwctrl, reward_function = (s,a,ss) -> 0.2*gwctrl.reward_function(s,a,ss) ) # softer control
    stronger_gwctrl = modify(gwctrl, reward_function = (s,a,ss) -> 0.8*gwctrl.reward_function(s,a,ss) ) # harder 
    ν = generate_ν(gwctrl)
    (_,furthest) = findmin(ν.dict)
    V_lo = generate_ν(weaker_gwctrl)
    V_hi = generate_ν(stronger_gwctrl)
    experiment_lo = Dict(β => learning_traj(gwctrl, V_lo; β, n = 5) for β in exp.(range(-6,0,6)))
    experiment_hi = Dict(β => learning_traj(gwctrl, V_hi; β, n = 5) for β in exp.(range(-6,0,6)))
    (tru = ν, lo = experiment_lo, hi = experiment_hi, randomness = 0.1)
end

let out = out
plot_experiment(out.lo, findmin(out.ν.dict)[2], out.tru; noise = out.randomness)
end

out2 = let gwctrl = gridworld_ctrl(gw; randomness = 0.01, 
    γ = 0.99)
    weaker_gwctrl = modify(gwctrl, reward_function = (s,a,ss) -> 0.2*gwctrl.reward_function(s,a,ss) ) # softer control
    stronger_gwctrl = modify(gwctrl, reward_function = (s,a,ss) -> 0.8*gwctrl.reward_function(s,a,ss) ) # harder 
    ν = generate_ν(gwctrl)
    (_,furthest) = findmin(ν.dict)
    V_lo = generate_ν(weaker_gwctrl)
    V_hi = generate_ν(stronger_gwctrl)
    experiment_lo = Dict(β => learning_traj(gwctrl, V_lo; β, n = 5) for β in exp.(range(-6,0,6)))
    experiment_hi = Dict(β => learning_traj(gwctrl, V_hi; β, n = 5) for β in exp.(range(-6,0,6)))
    (tru = ν, lo = experiment_lo, hi = experiment_hi, randomness = 0.01)
end

out3 = let gwctrl = gridworld_ctrl(gw; randomness = 0.0, 
    γ = 0.9999)
    weaker_gwctrl = modify(gwctrl, reward_function = (s,a,ss) -> 0.2*gwctrl.reward_function(s,a,ss) ) # softer control
    stronger_gwctrl = modify(gwctrl, reward_function = (s,a,ss) -> 0.8*gwctrl.reward_function(s,a,ss) ) # harder 
    ν = generate_ν(gwctrl)
    (_,furthest) = findmin(ν.dict)
    V_lo = generate_ν(weaker_gwctrl)
    experiment_lo = Dict(β => learning_traj(gwctrl, V_lo; β, n = 5) for β in exp.(range(-6,0,6)))
    (tru = ν, traj = experiment, randomness = 0.0, γ = γ, ctrl = gwctrl)
end

let out = out3
    plot_experiment(out.traj, findmin(out.tru.dict)[2], out.tru; noise = out.γ)
end



function plot_experiment(exp_dict, furthest, trueval; noise = .1)
    f = Figure()
    ax = Axis(f[1,1], xlabel = L"iteration $i$: $V^{i+1} \rightarrow V^{i} + β^{-1}\log Z^{β}$", ylabel = L"residual: $V^i(s_0) - \nu(s_0)$", title = "noise = $noise")
    trs(x) = sign(x)*abs(x)^(1/2)
    prs = sort(collect(pairs(exp_dict)),by = x->-x[1])
    l = length(prs) 
    i = 0
    for (β,d) in prs
        out = map(x -> trs( x(furthest)- trueval(furthest)), d)[2:end]
        lines!(ax,out; label = "β = $(round(β, digits = 3))",
        color = ColorSchemes.roma[i/(length(prs)-1)],
        linewidth = 2.0
        )
        i += 1
    end
    axislegend(ax)
    f
end

function plot_optimality_gap(gap_dict, furthest, trueval; noise = .1)
    f = Figure()
    ax = Axis(f[1,1], xlabel = L"iteration $i$: $V^{i+1} \rightarrow V^{i} + β^{-1}\log Z^{β}$", ylabel = L"optimality gap: $R^i(s_0) - \nu(s_0)$", title = "noise = $noise", yscale = log10)
    prs = sort(collect(pairs(gap_dict)),by = x->-x[1])
    l = length(prs) 
    i = 0
    for (β,d) in prs
        out = map(x ->  x(furthest), d)[2:end]
        lines!(ax,out; label = "β = $(round(β, digits = 3))",
        color = ColorSchemes.roma[i/(length(prs)-1)],
        linewidth = 2.0
        )
        i += 1
    end
    axislegend(ax)
    f
end




plot_experiment(experiment_lo,furthest,ν)
plot_experiment(experiment_hi,furthest,ν)




experiment_lo = Dict(β => learning_traj(V_lo; β, n = 5) for β in exp.(range(-6,0,6)))
experiment_hi = Dict(β => learning_traj(V_hi; β, n = 5) for β in exp.(range(-6,0,6)))


plot_experiment(experiment_lo,furthest,ν)
plot_experiment(experiment_hi,furthest,ν)




function run_experiment(gw; randomness = 0.01, γ = .99, 
    starting_temp = 0.1,
    βrange = exp.(range(-6,0,8)))
    #generates a named tuple
    gwctrl = gridworld_ctrl(gw; randomness, γ)

    weaker_gwctrl = modify(gwctrl, reward_function = (s,a,ss) -> starting_temp*gwctrl.reward_function(s,a,ss) )
    
    ν = generate_ν(gwctrl)
    (_,furthest) = findmin(ν.dict)
    V_lo = generate_ν(weaker_gwctrl)
    experiment = Dict(β => learning_traj(gwctrl, V_lo; β, n = 5)    
        for β in βrange)
    optimality_gap = Dict(β => 
            [generate_optimality_gap(gwctrl, v; ν) for v in vec]
        for (β,vec) in pairs(experiment))

    return (ν = ν, traj = experiment, gap = optimality_gap,
        randomness = randomness, γ = γ, ctrl = gwctrl, s0 = furthest)
end



function plot_experiment(exp_dict, furthest, trueval)
    f = Figure()
    ax = Axis(f[1,1], xlabel = "iteration (V → V + log(Z)/β)", ylabel = "V-residual (sqrt-scale)")
    trs(x) = sign(x)*abs(x)^(1/2)
    prs = sort(collect(pairs(exp_dict)),by = x->-x[1])
    l = length(prs) 
    i = 0
    for (β,d) in prs
        out = map(x -> trs( x(furthest)- trueval(furthest)), d)[2:end]
        lines!(ax,out; label = "β = $(round(β, digits = 3))",
        color = ColorSchemes.roma[i/length(prs)],
        linewidth = 2.0
        )
        i += 1
    end
    axislegend(ax)
    f
end

function learning_traj(v0; β = 1.0, n = 20)
    out = [v0]
    vnew = v0
    for _ = 1:n
        vnew = logz_updateV(gwctrl, vnew; β)
        push!(out,vnew)
    end
    return out
end




out1 = [begin
V_hi_up = logz_updateV(gwctrl, V_hi; β = exp(ii))
extrema(matrixify(V_hi_up) - matrixify(ν))
end for ii in range(0,-10,30)]

out2 = [begin
V_lo_up = logz_updateV(gwctrl, V_lo; β = exp(ii))
extrema(matrixify(V_lo_up) - matrixify(ν))
end for ii in range(0,-10,30)]

begin
    f = Figure()
    ax = Axis(f[1,1], xlabel = "β", ylabel = "residual range")
    band!(ax,range(0,-10,30),getindex.(out1,1), getindex.(out1,2))
    f
end

begin
f = Figure()
ax = Axis(f[1,1], xlabel = "β", ylabel = "residual range")
band!(ax,range(0,-10,30),getindex.(out2,1), getindex.(out2,2))
f
end

extrema(matrixify(V_hi0) - matrixify(ν))

function matrixify(g; filling = 0.0)
    s = maximum(k for k in keys(g.dict))
    m = fill(filling,Tuple(s) .+ 1)
    for k in keys(g.dict)
        m[k] = g(k)
    end
    return m
end

z_hi = generate_logz(gwctrl, V_hi; β = 0.01)
z_lo = generate_z(gwctrl, V_lo; β = 1.0)

matplot(ν)
gw.goal


matplot(x) = heatmap(matrixify(x; filling = NaN), colormap = :bluesreds, colorrange=(-4,4), colorscale = asinh, nan_color = RGB(.0))

heatmap(matrixify(V_hi))
heatmap(log.(matrixify(z_hi)))
heatmap(log.(matrixify(z_lo)))