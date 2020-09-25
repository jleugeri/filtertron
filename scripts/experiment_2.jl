using DrWatson
@quickactivate "Filtertron"

using Filtertron, DifferentialEquations, Flux, Optim, DiffEqFlux, DiffEqSensitivity, Plots
import ForwardDiff

struct Inputs{FT}
    tmp::Vector{Float64}
    f!::FT
end
Inputs(dims::Int, f!) = Inputs(zeros(Float64,dims), f!)
(i::Inputs)(t) = (i.f!(i.tmp, t); i.tmp)

num_inputs=1
num_neurons = 1
output_neuron = num_neurons # last neuron is output
num_taps = 10 
α = 10.0
σ = 0.1
input = Inputs(num_inputs, (x,t) -> (x .= sin(2t);nothing))
target(t) = sin(4t+0.5)

W_in_0 = randn(Float64, num_taps, num_inputs, num_neurons)
W_fw_0 = randn(Float64, num_taps, num_neurons, num_neurons)
W_fb_0 = randn(Float64, num_taps, num_neurons)
bias0  = randn(Float64, num_neurons)

raw_params = γNeurons(num_neurons, num_taps, input, W_in_0, W_fw_0, W_fb_0, bias0; α=α, σ=σ)
wrapped_params = DynamicWrapper(raw_params, (:W_in, :W_fw, :W_fb, :bias); RT=Union{Vector{Float64}, Vector{<:ForwardDiff.Dual}})

tspan = (0.0,10.0)
prob = FNNProblem(tspan, wrapped_params)

sol = solve(prob, SOSRI())

# Plot the solution
plot(sol, ylim = (-2, 2))

function find_solution(prob, wrapped_params; learning_rate=0.1, maxiters = 100, kwargs...)
    tsteps = LinRange(prob.tspan..., 101)
    function loss(p)
        sol = solve(prob, p=p, sensealg = ForwardDiffSensitivity(), saveat = tsteps)
        # wrapped_params(p)

        loss = sum(abs2, sol[wrapped_params.state[output_neuron][1],:] .- target.(tsteps))
        return loss, sol
    end

    function callback(p, l, sol)
        display(l)
        # sol = solve(prob, p=p, sensealg = ForwardDiffSensitivity(), saveat = tsteps)
        plt = plot(sol, ylim = (-2, 2))
        plot!(tsteps, target.(tsteps), color=:black, linestyle=:dash)
        display(plt)
        
        return false
    end

    result_ode = DiffEqFlux.sciml_train(loss, prob.p, ADAM(learning_rate), cb = callback, maxiters=maxiters, kwargs...)
    return result_ode
end

res=find_solution(prob, wrapped_params, maxiters=1000, learning_rate=0.01)
