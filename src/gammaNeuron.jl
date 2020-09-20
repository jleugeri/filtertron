using Parameters, RecursiveArrayTools
export γNeurons

const soma_id = 1

struct γNeurons{WINT,WFWT,WFBT,BT,IT,UT,FT}
    num_neurons::Int            # number of neurons
    num_taps::Int               # number of taps per neuron
    input::IT                   # external input function `(t,nid)->input` that the neuron `id` is driven with at time `t`
    W_in::WINT                  # input weights
    W_fw::WFWT                  # weights between the neurons
    W_fb::WFBT                  # feed-back weights within the neurons
    bias::BT                    # bias weights within the neurons
    α::Float64                  # filter decay rate
    σ::Float64                  # noise incurred per filter tap per neuron
    f::FT                       # nonlinearity
    u0::UT                      # initial state vector
    state::Vector{UnitRange{Int64}} # indices of state variables
end

function γNeurons(num_neurons, num_taps, input, W_in, W_fw, W_fb=zeros(Float64,num_taps, num_neurons), bias=zeros(Float64,num_neurons); α=10.0, σ=0.1, f=tanh, u0=randn(Float64, num_taps*num_neurons).*σ)
    state = [((i-1)*num_taps+1):((i)*num_taps) for i ∈ 1:num_neurons]
    return γNeurons(num_neurons, num_taps, input, W_in, W_fw, W_fb, bias, α, σ, f, u0, state)
end


function ∇!(du, u, p::DynamicWrapper{<:γNeurons}, t, i) where T
    du[p.state[i]] .= p.α .* ((p.W_in[:,:,i] * p.input(t)) .- u[p.state[i]])
    du[p.state[i][1:end-1]] .+= p.α .* u[p.state[i][2:end]]

    du .+= p.α .* tanh.(p.W_fb[:,i])./p.num_taps * u[p.state[i][soma_id]]
    nothing
end

function couple!(du, u, p::DynamicWrapper{<:γNeurons}, t, i) where T
    out = p.f(u[p.state[i][soma_id]]+p.bias[i])
    for (j,idx) ∈ enumerate(p.state)
        du[idx] .+= p.α .* out .* p.W_fw[:,j,i]
    end
    nothing
end

function η!(du, u, p::DynamicWrapper{<:γNeurons}, t, i)
    du[p.state[i]] .= p.σ 
    nothing
end
