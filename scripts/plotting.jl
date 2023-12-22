##
using CairoMakie
using ColorSchemes
using PIQL
using Colors


 
s = (8,8)
gw = make_gridworld(s; density = 0.2)
gwctrl = gridworld_ctrl(gw; randomness = 0.01, γ = 0.99, reward_scale = 1.0, reward = 2.0)
weaker_gwctrl = modify(gwctrl, reward_function = (s,a,ss) -> 0.6*gwctrl.reward_function(s,a,ss) ) # softer control
stronger_gwctrl = modify(gwctrl, reward_function = (s,a,ss) -> 1.4*gwctrl.reward_function(s,a,ss) ) # harder 

ν = generate_ν(gwctrl)
V_lo = generate_ν(weaker_gwctrl)
V_hi = generate_ν(stronger_gwctrl)


V_hi0 = z_updateV(gwctrl, V_hi; β = 0.1)
V_lo0 = z_updateV(gwctrl, V_lo; β = 0.1)
extrema(matrixify(V_lo0) - matrixify(ν))


extrema(matrixify(ν))

out = [begin
V_lo0 = z_updateV(gwctrl, V_lo; β = exp(ii))
extrema(matrixify(V_lo0) - matrixify(ν))
end for ii in range(0,-7,30)]

out2 = [begin
V_lo_up = logz_updateV(gwctrl, V_lo; β = exp(ii))
extrema(matrixify(V_lo_up) - matrixify(ν))
end for ii in range(0,-6,30)]

begin
f = Figure()
ax = Axis(f[1,1], xlabel = "β", ylabel = "residual range")
band!(ax,exp.(range(0,-6,30)),getindex.(out2,1), getindex.(out2,2))
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