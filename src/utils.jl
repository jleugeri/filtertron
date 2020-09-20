export DynamicWrapper, fieldnames, getproperty, getfield

struct DynamicWrapper{T,ST,DT,RT}
    static_fields::ST
    dynamic_fields::DT
    store::Ref{RT}
end

function DynamicWrapper(obj::T, dynamic_params; store_type=Vector{Float64}, RT=store_type) where T
    static_fields = Pair{Symbol,Any}[]
    dynamic_fields = Pair{Symbol,Any}[]
    store = store_type()
    offset = 1
    for fieldname ∈ fieldnames(T)
        if fieldname ∈ dynamic_params
            old_val = getfield(obj, fieldname)
            next_offset = offset + length(old_val)
            push!(dynamic_fields, fieldname => reshape(offset:(next_offset-1),size(old_val)))
            if ndims(old_val) == 0
                push!(store, old_val)
            else
                append!(store, old_val[:])
            end
            offset = next_offset
        else
            push!(static_fields, fieldname => getfield(obj, fieldname))
        end
    end
    sf = (; static_fields...)
    df = (; dynamic_fields...)
    st = Ref{RT}(store)
    return DynamicWrapper{T, typeof(sf), typeof(df), RT}(sf,df,st)
end

(Base.fieldnames(::Type{DynamicWrapper{T,ST,DT}}) where {T,ST,DT}) = fieldnames(T)

(dw::DynamicWrapper)(store) = (getfield(dw,:store)[] = store; dw)
(Base.getproperty(dw::DynamicWrapper{T,ST,DT}, sym::Symbol) where {T,ST,DT}) = if sym ∈ fieldnames(ST)
    getfield(getfield(dw,:static_fields), sym)
else
    view(getfield(dw,:store)[], getfield(getfield(dw,:dynamic_fields), sym))
end