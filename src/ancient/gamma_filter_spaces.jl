using LinearAlgebra, AbstractAlgebra, SpecialMatrices, RecursiveArrayTools, Distributions

export GammaFilter, GammaFilterRing, gram, ContinuousSignal, EventBasedSignal, Signal, Filter, apply_filters

#= Filters and Spaces =#
abstract type Signal end
abstract type Filter{T} <: AbstractAlgebra.RingElem end
abstract type FilterRing <: AbstractAlgebra.Ring end

#= Defining GammaFilters =#
mutable struct GammaFilterRing{T} <: FilterRing
    α::T
    _G::Symmetric{T,Matrix{T}}
    GammaFilterRing{T}(α::T) where T<:Real = new{T}(α, Symmetric(Matrix{T}(undef, 0, 0)))
end

struct GammaFilter{T} <: Filter{T}
    coefficients::Vector{T}
    space::GammaFilterRing{T}
end


#= special elements =#
Base.zero(R::GammaFilterRing{T}) where T = GammaFilter(T[],R)
Base.one(R::GammaFilterRing{T}) where T = GammaFilter([T(1)],R)
AbstractAlgebra.iszero(f::GammaFilter{T}) where T = T[]==f.coefficients
AbstractAlgebra.isone(f::GammaFilter{T}) where T = T[1]==f.coefficients

#= canonicalisation =#
#TODO: implement!
AbstractAlgebra.canonical_unit(f::GammaFilter) = one(parent(f))

#= IO =#
#TODO: improve IO
show(io::IO, R::GammaFilterRing) = show(io, "GammaFilterRing(α=$(R.α))")
show(io::IO, f::GammaFilter) = show(io, "GammaFilter($(f.coefficients),$(parent(f)))")
AbstractAlgebra.needs_parentheses(f::GammaFilter) = false
AbstractAlgebra.displayed_with_minus_in_front(f::GammaFilter) = false
AbstractAlgebra.show_minus_one(::Type{GammaFilter}) = false

#= operations =#
# helper function
function apply_with_default(op, c...;default=0)
    c = [DefaultArray(cc,default) for cc ∈ c]
    final_size=promote_size(c...)
    idx=Base.OneTo.(final_size)
    broadcast(op, (cc[idx...] for cc ∈ c)...)
end

(Base.:-)(f::GammaFilter) = -1*f
(Base.:+)(op1::GammaFilter,op2::GammaFilter) = GammaFilter(apply_with_default(+,op1.coefficients,op2.coefficients), op1.space)
(Base.:-)(op1::GammaFilter,op2::GammaFilter) = GammaFilter(apply_with_default(-,op1.coefficients,op2.coefficients), op1.space)
(Base.:*)(op1::GammaFilter,op2::GammaFilter) = GammaFilter([
    reduce(
        +,
        op1.coefficients[k] * op2.coefficients[i+1-k] for k ∈ 1:length(op1.coefficients) if 1<=i+1-k<=length(op2.coefficients);
        init=0.0
    ) for i ∈ 1:length(op1.coefficients)+length(op2.coefficients)-1
], op1.space)

(Base.:^)(f::GammaFilter, e::Int) =
    if e == 0
        zero(parent(f))
    elseif e == 1
        e
    elseif e > 1
        res = f
        for i ∈ 1:e-1
            res = res*f
        end
        res
    else
        DomainError("Cannot take power $(e) of GammaFilter!")
    end

#= comparisons =#
(Base.:(==))(f::GammaFilter, g::GammaFilter) = all(apply_with_default(==, f.coefficients, g.coefficients)) && f.space==g.space
Base.isequal(f::GammaFilter, g::GammaFilter) = f==g
Base.isapprox(f::GammaFilter, g::GammaFilter; atol::Real=sqrt(eps())) = all(apply_with_default((a,b)->isapprox(a,b;atol=atol), f.coefficients, g.coefficients))

# TODO: return exact division if possible
AbstractAlgebra.divexact(f::GammaFilter, g::GammaFilter) = error("Cannot do exact division between GammaFilters atm.")

#= additional constructors =#
(R::GammaFilterRing)() = zero(R)
(R::GammaFilterRing{T})(a::Integer) where T = GammaFilter([T(a)],R)
(R::GammaFilterRing)(f::GammaFilter) = R==parent(f) ? f : error("$(f) is not element of this ring ($(R))!")

#= Define parent - element relationship between ring and elements =#
AbstractAlgebra.parent_type(::Type{GammaFilter{T}}) where T = GammaFilterRing{T}
AbstractAlgebra.elem_type(::Type{GammaFilterRing{T}}) where T = GammaFilter{T}
# TODO: parameterized by base ring
AbstractAlgebra.base_ring(R::GammaFilterRing{T}) where T = Union{}
AbstractAlgebra.parent(f::GammaFilter{T}) where T = f.space

#= Define properties of the ring =#
#TODO: check if parent is integral domain
AbstractAlgebra.isdomain_type(::Type{GammaFilter{T}}) where T = false
#TODO: check if is exact (e.g. T==Int)
AbstractAlgebra.isexact_type(::Type{GammaFilter{T}}) where T = false

#= deepcopy =#
AbstractAlgebra.deepcopy_internal(f::GammaFilter{T}, ::IdDict) where T = GammaFilter{T}(deepcopy(f.coefficients), f.space)

#= mutating operators =#
# TODO: actual performance improvements
AbstractAlgebra.zero!(f::GammaFilter) = (empty!(f.coefficients);f)
AbstractAlgebra.mul!(c::GammaFilter,a::GammaFilter,b::GammaFilter) = a*b
AbstractAlgebra.add!(c::GammaFilter,a::GammaFilter,b::GammaFilter) = a+b
AbstractAlgebra.addeq!(b::GammaFilter,a::GammaFilter) = a+b

#= random generation =#
#TODO: rand(R::GammaFilterRing,v...) = ...

#= Promotion rules =#
promote_rule(::Type{GammaFilter{T}}, ::Type{GammaFilter{T}}) where T = GammaFilter{T}
function promote_rule(::Type{GammaFilter{T}}, ::Type{U}) where {T,U}
       promote_rule(T, U) == T ? GammaFilter{T} : Union{}
end

function apply_filters(ops::Matrix{<:GammaFilter}, signals::Matrix{<:Signal})
    num_taps = maximum(op->length(op.coefficients), ops; dims=2)

    function f′_individual(du,u,α,inp,t)
        @. du[1:end-1] += α * (u[2:end] - u[1:end-1] + inp[1:end-1])
        du[end] += -α * (u[end] + inp[end])
    end
    
    function f′(du, u, p, t)
        l=1
        for i ∈ 1:size(ops,1)
            for k ∈ 1:size(signals, 2)
                du.x[l] .= zero(eltype(ops[i,k].coefficients))
                for j ∈ 1:size(ops,2)
                    inp = isa(signals[j,k], ContinuousSignal) ? signals[j,k](t) * ops[i,j].coefficients : zeros(eltype(ops[i,j].coefficients),length(ops[i,j].coefficients))
                    f′_individual(du.x[l], u.x[l], ops[i,j].space.α, inp, t)
                end
                l+=1
            end
        end
    end
    
    tspan_min = minimum(s->s.tspan[1], signals)
    tspan_max = maximum(s->s.tspan[2], signals)
    tspan = (tspan_min, tspan_max)
    
    event_queue = sort!([(t,(i,j)) for i ∈ 1:size(signals,1) for j ∈ 1:size(signals,2) if isa(signals[i], EventBasedSignal) for t ∈ signals[i].times], by=first)
    tstops = first.(event_queue)
    check_spike(u,t,integrator) =  ~isempty(event_queue) && event_queue[1][1] <= t
    
    function spike_effect!(integrator)
        (_,(i,j)) = popfirst!(event_queue)
        for k ∈ 1:size(ops,1)
            l=(j-1)*size(ops,1)+k
            integrator.u.x[l] .+= ops[k,i].space.α .* ops[k,i].coefficients
        end
        u_modified!(integrator, true)
    end

    tp = eltype(ops[1].coefficients)
    u₀ = ArrayPartition((zeros(tp, num_taps[i]) for i∈1:size(ops,1) for j∈1:size(signals,2))...)
    cb = DiscreteCallback(check_spike, spike_effect!; save_positions=(true,true))
    prob = ODEProblem(f′, u₀, tspan, nothing)
    sol = DifferentialEquations.solve(prob, Tsit5(); callback=cb, tstops=tstops)
    
    # return t->reshape(collect(first.(sol(t).x)), (size(ops,1), size(signals,2)))
    return reshape([ContinuousSignal(t->sol(t).x[i][1],tspan) for i ∈ eachindex(u₀.x)],(size(ops,1), size(signals,2)))
end
# (Base.:*)(ops::AbstractVector{<:GammaFilterOperator},signals::AbstractVector{<:Signal}) = signals*ops
# (Base.:*)(signal::Signal, op::GammaFilterOperator) = first∘([signal]*[op])
# (Base.:*)(op::GammaFilterOperator, signal::Signal) = signal*op



#=


struct GammaFilterOperator{T,C} <: Filter{T}
    coefficients::C
    space::GammaFilterRing{T}
end

struct IdentityFilterOperator{T} <: Filter{T}
end

struct ZeroFilterOperator{T} <: Filter{T}
end

(Base.one(::Type{<:Filter{T}})) where T = IdentityFilterOperator{T}()
(Base.one(op::Filter{T})) where T = IdentityFilterOperator{T}()
(Base.oneunit(::Type{<:Filter{T}})) where T = IdentityFilterOperator{T}()
(Base.oneunit(op::Filter{T})) where T = IdentityFilterOperator{T}()
(Base.zero(::Type{<:Filter{T}})) where T = ZeroFilterOperator{T}()
(Base.zero(op::Filter{T})) where T = ZeroFilterOperator{T}()

(Base.:*)(op1::IdentityFilterOperator,op2) = op2
(Base.:*)(op1::ZeroFilterOperator,op2) = op1
(Base.:*)(op1::GammaFilterOperator,op2::GammaFilterOperator) = GammaFilterOperator([
    reduce(
        +,
        op1.coefficients[k] * op2.coefficients[i-k] for k ∈ 1:length(op1.coefficients)+length(op2.coefficients) if 1<=k<=length(op1.coefficients) && 1<=i-k<=length(op2.coefficients);
        init=0.0
    ) for i ∈ 1:length(op1.coefficients)+length(op2.coefficients)
], op1.space)


"""
Solve the deconvolution Problem

minimize ||s(t) * ∑wᵢfᵢ(t) * ∑vᵢfᵢ(t) - s(t)*∑uᵢfᵢ(t)||

where s is a signal, f are basis functions with Gram-matrix F and wᵢ,vᵢ and uᵢ
are the coefficients of the signal autocorrelation, deconvolution filter and desired final impulse response, respectively.

Returns the vector (vᵢ)ᵢ.
"""
function (Base.:\)(w::GammaFilterOperator{T},u::GammaFilterOperator{T}) where T
    @assert w.space === u.space "Spaces must coincide"
    N = length(w.coefficients)
    M = length(u.coefficients)
    
    # construct Toeplitz matrix W for convolution with w
    W = Toeplitz(vcat(zeros(T, M), w.coefficients, zeros(T, M-N-1)))[:,1:M-N]

    # construct geometry matrix
    FW = gram(w.space, 1:size(W,1), 1:size(W,1))*W
    M = (W'*FW)
    u′= FW'*u.coefficients

    # solve
    v = M\u′
    return GammaFilterOperator(v, w.space)
end


(B
(Base.:*)(a::Number, op1::GammaFilterOperator) = GammaFilterOperator(op1.coefficients.*a, op1.space)
(Base.:*)(op1::GammaFilterOperator,a::Number) = a*op1
(Base.:+)(op1::GammaFilterOperator,op2::GammaFilterOperator) = GammaFilterOperator(op1.coefficients+op2.coefficients, op1.space)
(Base.:+)(op1::Filter,op2::ZeroFilterOperator) = op1
(Base.:+)(op1::ZeroFilterOperator,op2::Filter) = op2
(Base.:-)(op1::GammaFilterOperator,op2::GammaFilterOperator) = GammaFilterOperator(op1.coefficients-op2.coefficients, op1.space)
(Base.:-)(op1::GammaFilterOperator) = (-1)*op1
=#