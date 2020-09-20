using StochasticDiffEq
export FNNProblem, η!, ∇!, couple!

"""
    ∇!(du.x[i], u.x[i], p, t, i)
Deterministic (drift) part of the equation(s) for the individual neuron"""
function ∇! end;

"""
    couple!(du.x, u.x[i], p, t, i)
Coupling of neuron `i` to other neurons' state equations"""
function couple! end;

"""
    η!(du.x[i], u.x[i], p, t, i)
Stochastic (diffusion) part of the equation(s) for the individual neuron"""
function η! end;


"""
    FNNProblem(tspan, p; kwargs...)

Set up an FNN simulation as a `SDEProblem`.
Returns a function `run(kwargs...)` which produces the solution of the `SDEProblem` (passing along the parameters `kwargs`).

# Arguments:
- `tspan::Tuple{Float64,Float64}`: tuple `(start,end)` defining the time-span over which to run the simulation
- `p`:  parameters
    - `p.u0::Vector{Float64}`: vector of initial conditions for the neurons' state variables
"""
function FNNProblem(tspan, param_wrapper::DynamicWrapper; kwargs...)
    p0 = getfield(param_wrapper, :store)[]

    """
    ∇FNN(du, u, p, t)
    """
    function ∇FNN!(du, u, p, t)
        param_wrapper(p)
        for i ∈ 1:param_wrapper.num_neurons
            ∇!(du, u, param_wrapper, t, i)
            couple!(du, u, param_wrapper, t, i)
        end
        nothing
    end

    """
    η!(du, u, p, t)
    """
    function ηFNN!(du, u, p, t)
        param_wrapper(p)
        for i ∈ 1:param_wrapper.num_neurons
            η!(du, u, param_wrapper, t, i)
        end
        nothing
    end

    SDEProblem(∇FNN!, ηFNN!, param_wrapper.u0, tspan, p0; kwargs...)
end
