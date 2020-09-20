using DifferentialEquations, DiffEqFlux, Flux, Plots

f(u, w, t) = [0.0, w' * u]
f_flux(u, w, t) = Tracker.collect([0.0, sum(w .* u)])

u₀ = [0.0, 0.0]
w = [0.0, 1.0]
p = param(w)
tspan = (0.0,100.0)

prob = ODEProblem(f, u₀, tspan, w)
prob_flux = ODEProblem(f_flux, u₀, tspan, w)

diffeq_rd(p,prob_flux)

function predict_rd()
    diffeq_rd(p,prob_flux,Tsit5(),saveat=0.1)[1,:]
end
loss_rd() = sum(abs2,x-1 for x in predict_rd())

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
# cb = function ()
#     display(loss_rd())
#     display(plot(solve(remake(prob,p=Flux.data(p)),Tsit5(),saveat=0.1),ylim=(0,6)))
# end

cb()
Flux.train!(loss_rd, [p], data, opt, cb = cb)

using Flux
x = param([1.0,2.0])
y = [1.0,2.0]
f1(x, y) = Tracker.collect([0.0, sum(x .* y)])
f2(x, y) = Tracker.collect([0.0; sum(x .* y)])
f3(x, y) = Tracker.collect([0.0, sum(x .* y)])

println(f1(x,y))
println(f1(x,y))

Flux.back!(sum(f2(x,y)))
#Flux.back!(sum(f1(x,y)))
