using DifferentialEquations, Distributions, Plots, thesis

thesis.init()

function f′(du, u, p, t)
    du .= -p.α .* u
    du[2:end] .+= p.α*u[1:end-1]
end

p = (
    α = 10.0,
    δ = 1.0,
    spiketimes = Float64[],
    weights = [0.0, 1.5, -1.0, -2.0, 2.5]
)
tspan = (0.0,10.0)
tspan_inset = (4,6)
tspan_impulse = (-0.1,1.0)
input_spikes = sort(rand(rand(Poisson(5.0*(tspan[2]-tspan[1])))).*(tspan[2]-tspan[1]).+tspan[1])
u₀ = zeros(5)


spike_threshold(u,t,integrator) = u'*integrator.p.weights - integrator.p.δ
function spike_reset!(integrator)
    integrator.u .= 0
    u_modified!(integrator, true)
    push!(integrator.p.spiketimes, integrator.t)
end

function spike_effect!(integrator)
    integrator.u[1]+=1;
    if spike_threshold(integrator.u, integrator.t, integrator) >= 0.0
        spike_reset!(integrator)
    end
    u_modified!(integrator, true)
end

cb = CallbackSet(DiscreteCallback((u,t,integrator) -> t ∈ input_spikes, spike_effect!; save_positions=(true,true)), ContinuousCallback(spike_threshold, spike_reset!, nothing; save_positions=(true,true)))

sol = solve(ODEProblem(f′, u₀, tspan.-(10.0,0), p); callback=cb, tstops=input_spikes)
state(t) = sol(t)'*p.weights

test_spike=[0.0]
cb = CallbackSet(DiscreteCallback((u,t,integrator) -> t ∈ test_spike, spike_effect!; save_positions=(true,true)), ContinuousCallback(spike_threshold, spike_reset!, nothing; save_positions=(true,true)))
sol′ = solve(ODEProblem(f′, u₀, tspan_impulse, p); callback=cb, tstops=test_spike)
impulse_response(t) = sol′(t)'*p.weights


spiketrain = filter(s->tspan[1]<=s<=tspan[2], p.spiketimes)

p_in = plot([tspan_inset...], [1.1,1.1], fill=t->-0.1, color="#fcaf3e", alpha=0.5, ylims=(-0.1,1.1))
vline!(input_spikes, color="#babdb6ff", label="input spikes")
plot!(state, tspan..., xlims=tspan.+(0,0.2), xlabel=false, color=:black, ylabel="state", label="state ")
hline!([p.δ], color=thesis_colors[end], linestyle=:dash, label="threshold", yticks=[], xlims=tspan.+(0,0.5))
scatter!(spiketrain, fill(p.δ,length(spiketrain)), markersize=3, markercolor=:black, markerstrokewidth=2, markerstrokecolor=thesis_colors[end], label="")
plot!([spiketrain'; spiketrain'], [fill(0,length(spiketrain))';fill(p.δ,length(spiketrain))'], linewidth=2, color=thesis_colors[end], label="")


input_spikes_inset = filter(s->tspan_inset[1]<=s<=tspan_inset[2], input_spikes)
p_in_inset = plot([tspan_inset...], [2.0, 2.0], fill=0, color="#fcaf3e", alpha=0.5, label="")
vline!(input_spikes_inset, color="#babdb6ff", label="")
plot!(sol, tspan=tspan_inset, label="", ylabel="state ", color=thesis_colors')
plot!(state, tspan_inset..., xlims=tspan_inset, color=:black, xlabel=false, label="")
hline!([p.δ], color=thesis_colors[end], linestyle=:dash, label="", yticks=[])

p_impulse_inset = plot(sol′, tspan=tspan_impulse, label="", title="impulse response ")
vline!(test_spike, color="#babdb6ff", label="")
plot!(impulse_response, tspan_impulse..., xlims=tspan_impulse, color=:black, xlabel=false, label="")
hline!([p.δ], color=thesis_colors[end], linestyle=:dash, label="", yticks=[])


p_spike = plot(xlabel="time", ylabel="output")
plot!(p_spike, [spiketrain'; spiketrain'], [fill(1,length(spiketrain))';fill(1+0.75,length(spiketrain))'], linewidth=2, color=thesis_colors[end], label="")
scatter!(p_spike, spiketrain, fill(1+0.75,length(spiketrain)), markersize=3,  markercolor=:black, markerstrokewidth=2, markerstrokecolor=thesis_colors[end], label="", yticks=[], ylims=(1,2.1), xlims=tspan.+(0,0.5))

p_insets = plot(p_in_inset, p_impulse_inset, layout=grid(1,2))

plot(p_insets, p_in, p_spike, layout=grid(3,1, heights=[0.4, 0.4, 0.2], link=:y), legend=false)

savefig("figures/filtertron.pdf")
savefig("figures/filtertron.svg")
