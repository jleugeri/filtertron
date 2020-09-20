using Plots,DifferentialEquations, DiffEqFlux, LabelledArrays, Flux, LinearAlgebra
#thesis.init()

const tspan = (0.0,500.0)

pat1(τ) = exp(-0.1*τ^2)*sin(τ)
pat2(τ) = exp(-0.1*τ^2) - exp(-0.1*(τ-5.0)^2)

t1 = rand(25) .* (tspan[2]-tspan[1]) .+ tspan[1]
t2 = rand(25) .* (tspan[2]-tspan[1]) .+ tspan[1]

inp1(t) = sum(pat1, t .- t1)
inp2(t) = sum(pat2, t .- t2)

const input = t->inp1(t)+inp2(t) #t->0.5*sin(2*t)+sin(t+1)+2.0*sin(4*t+2)
const target = inp1 #t->(t % 2π)-π#sin(t)
const N = 10
const α = 1.0
const β = 1.0
const δ = 1.0

u₀ = zeros(4N)

f_dec′(u, w, t) = (tmp1 = sum(w[1:2N] .* u[1:2N]); tmp2 = sum(w[2N+1:4N] .* u[2N+1:4N]); [
    input(t)::Float64,
    (α .* u[1:N-1])...,
    tmp1, #hacking the dot product in here because w'*u is `ambiguous`...
    (α .* u[N+1:2N-1])...,
    tanh(tmp1),
    (α .* u[2N+1:3N-1])...,
    tmp2, #hacking the dot product in here because w'*u is `ambiguous`...
    (α .* u[3N+1:4N-1])...
] .-α .* u)

f_dec_flux′(u, w, t) = (tmp1 = sum(w[1:2N] .* u[1:2N]); tmp2 = sum(w[2N+1:4N] .* u[2N+1:4N]); Tracker.collect([
    input(t)::Float64,
    (α .* u[1:N-1])...,
    tmp1, #hacking the dot product in here because w'*u is `ambiguous`...
    (α .* u[N+1:2N-1])...,
    tanh(tmp1),
    (α .* u[2N+1:3N-1])...,
    tmp2, #hacking the dot product in here because w'*u is `ambiguous`...
    (α .* u[3N+1:4N-1])...
]) .-α .* u)


t = LinRange(tspan..., 1000)
w = fill(-1.0,(4N,))
prob = ODEProblem(f_dec′, u₀, tspan, w)
prob_flux = ODEProblem(f_dec_flux′, u₀, tspan, w)

# p = param(w)
# params = Flux.Params([p])
p = params(w)

function predict_rd() # Our 1-layer neural network
    diffeq_rd(p,prob_flux,Tsit5(),saveat=t)[3N+1,:]
end

loss_rd() = √(sum(abs2, predict_rd().-target.(t))/length(target.(t)))


data = Iterators.repeated((), 2000)
opt = ADAM(0.01)
step = 1
function cb() #callback function to observe training
    global step += 1
    display(loss_rd())
    #   # using `remake` to re-create our `prob` with current parameters `p`
    sol=solve(remake(prob,p=p[1]),Tsit5(),saveat=t)
    x=sol[3N+1,:]
    plt=plot(t, input.(t), linestyle=:dash, label="input")
    plot!(t, target.(t), linestyle=:dash, label="target")
    plot!(t, x,label="solution at step $(step)")
    display(plt)

    # re-randomize in- and output
    t1 .= rand(25) .* (tspan[2]-tspan[1]) .+ tspan[1]
    t2 .= rand(25) .* (tspan[2]-tspan[1]) .+ tspan[1]
end
# Display the ODE with the initial parameter values.
cb()
Flux.train!(loss_rd, p, data, opt, cb=cb)
