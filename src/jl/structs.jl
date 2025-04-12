using UnPack, FFTW, AbstractFFTs, Random, JLD2, HDF5
using Printf, Dates, EllipsisNotation
using LinearAlgebra: mul!
import Base.show

 
###################
# Data structures #
###################

"""
Contains parameters defining a d-dim system
"""
struct System{d}
    d::Int;
    N::Int;         # Number of lattice points
    L::Float64;     # Size of system
    Δk::Float64;
    Δx::Float64;
    Δt::Float64;
    T::Float64;     # Temperature
    function System(d, N, L, Δt; T=0.)
        @assert N%2 == 0 
        Δx = L/N
        Δk = 2π/L
        new{d}(d, N, L, Δk, Δx, Δt, T)
    end
end


"""
Construc p² for a d-dim system
"""
function get_p2(sys::System)
    @unpack Δx, Δk, N, d = sys
    sys
    p_1 = collect(rfftfreq(N , N * Δk))
    p2 = @. p_1^2
    for i in 2:d
        shape = vcat([1 for _ in 2:i], [N])
        p_i = reshape(collect(fftfreq(N, N * Δk)), shape...)
        p2 = @. p2 + p_i^2
    end
    return p2
end


"""
Construc abs_p and x for a d-dim system
"""
function get_xp(sys::System)
    @unpack Δx, Δk, N, d = sys
    x = Vector{Vector{Float64}}()
    p = Vector{Vector{Float64}}()

    push!(x, [0:(N - 1);]*Δx)
    push!(p, [rfftfreq(N , N * Δk);])

    for i in 2:d
        push!(x, [0:(N - 1);]*Δx)
        push!(p, [fftfreq(N, N * Δk);])
    end
    return x, p
end


"""
Type of exponential time step.
Continas constants for time stepping.
time step fucntions are overloaded.
"""
abstract type TimeStep end

struct ETD1{d} <: TimeStep
    K1::Array{Float64, d};
    K2::Array{Float64, d};
    σ::Array{Float64, d};
end

struct ETD2O{d, N} <: TimeStep
    K1::Array{Any};
    K2::Array{Any};
    σ::Array{Any};
end

struct ETD2{d} <: TimeStep
    K1::Array{Float64, d};
    K2::Array{Float64, d};
    K3::Array{Float64, d};
    σ::Array{Float64, d};
end

function ETD1(sys::System, conserved::Bool, c_func::Function)
    @unpack d, Δt, N, L, T = sys
    p2 = get_p2(sys)
    c = @. c_func(p2)
    if conserved; @. c = p2 * c end
    
    ch = @. c*Δt
    K1 = @. exp(ch)
    K2 = @. (expm1(ch) / ch) * Δt
    if c[1]==0; K2[1] = Δt end

    σ = noise(sys, c, p2, conserved)
    
    return ETD1{d}(K1, K2, σ)
end

function ETD2O(sys::System, conserved::Bool, c_func::Function; N=2)
    @unpack d, Δt, N, L, T = sys
    p2 = get_p2(sys)
    c = @. c_func(p2)
    if conserved; @. c = p2 * c end

    K11 = 1.
    K21 = Δt
    σ1  = 0.
    
    ch2 = @. c*Δt
    K12 = @. exp(ch2)
    K22 = @. (expm1(ch2) / ch2) * Δt
    if c[1]==0; K22[1] = Δt end

    σ2 = noise(sys, c, p2, conserved)

    σ = [σ1, σ1, σ2, σ2]
    K1 = [K11, K11, K12, K12]
    K2 = [K21, K21, K22, K22]
    
    return ETD2O{d, N}(K1, K2, σ)
end

function ETD2(sys::System, conserved::Bool, c_func::Function)
    @unpack d, Δt, N, L, T = sys
    p2 = get_p2(sys)
    c = @. c_func(p2)
    if conserved; @. c = p2 * c end

    ch = @. c*Δt
    K1 = @. exp(ch)
    K2 = @. ((ch*exp(ch) + expm1(ch) - 2*ch) / ch^2) * Δt
    K3 = @. ((ch - expm1(ch))/ch^2) * Δt

    if c[1]==0; K2[1] = 3/2 * Δt; K3[1] = -1/2 * Δt end

    σ = noise(sys, c, p2, conserved)
    
    return ETD2{d}(K1, K2, K3, σ)
end

function noise(sys::System, c::Array{Float64}, p2::Array{Float64}, conserved::Bool)
    @unpack d, Δt, N, L, T = sys
    σ = @. sqrt( 2*T*(N^2/L)^d * expm1(2*c*Δt)/(2*c))
    if c[1]==0; σ[1] = sqrt( 2*T*(N^2/L)^d * Δt ) end
    if conserved; @. σ = sqrt(p2) * σ end
    return σ
end

 
"""
Contains tools used by the integrators for a d-dim system with real fields
"""
struct Tools{d, Tfplan, Tfplan_full, Tbplan, Trng}
    sys::System{d};
    x::Vector{Vector{Float64}};
    p::Vector{Vector{Float64}};
    p2::Array{Float64, d};
    step::TimeStep;
    fplan::Tfplan;
    fplan_full::Tfplan_full;
    bplan::Tbplan;
    rng::Trng;
    pAA::Float64;
    conserved::Bool;
    seed::Int;
end

function Tools(sys::System, seed::Int, conserved::Bool, step::TimeStep)
    @unpack d, N, L, Δt, T = sys
    
    x, p = get_xp(sys)
    p2 = get_p2(sys)
    pAA = maximum(p[1]) * 2/3 # Maximum p for any of the coordinates

    #TODO: struct for fourier tools
    dims = [N for _ in 1:d]
    ohr = zeros(Float64, dims...)
    fplan = plan_rfft(ohr)
    fplan_full = plan_fft(ohr)
    ohf = fplan*ohr
    bplan = plan_irfft(ohf, dims[1])

    rng = Random.Xoshiro(seed)

    Tools{d, typeof(fplan), typeof(fplan_full), typeof(bplan), typeof(rng)}(
        sys, x, p, p2, step, fplan, fplan_full, bplan, rng, pAA, conserved, seed
    )
end

function Tools(sys::System, seed::Int, conserved::Bool, time_step::F, c_func::Function) where F
    step = time_step(sys, conserved, c_func)
    Tools(sys, seed, conserved, step)
end

function Tools(sys; seed=1, conserved=true, time_step=ETD1, c_func=(p2->-p2))
    Tools(sys, seed, conserved, time_step, c_func)
end

show(io::IO, tool::Tools) = show(io, "Tools for $(tool.sys)")


"""
Contains a single d-dim field and its Fourier transform
"""
mutable struct Field{d}
    x::Array{Float64, d};
    k::Array{ComplexF64, d};
    conserved::Bool
end

const FieldArray{d} = Array{Field{d}, 1}
const Fields{d} = Union{Field{d}, FieldArray{d}}
function Base.getindex(A::Field, ::Nothing) return A end
function Base.copy(f::Field) return Field(f.x, f.k, f.tools) end

function Field(tools::Tools, fx::Array{Float64})
    @unpack conserved = tools
    @unpack d = tools.sys
    fk = tools.fplan * fx
    Field{d}(fx, fk, conserved)
end

function Field(tools::Tools,)
    @unpack N, d = tools.sys
    fx = zeros([N for _ in 1:d]...)
    Field(tools, fx)
end


#############
# Functions #
#############

function FieldArray(tools::Tools, fx::Array{Float64}, n::Int,)
    @unpack d = tools.sys
    A = FieldArray{d}()
    for i in 1:n
        push!(A, Field(tools, fx[i, ..]))
    end
    return A
end

function FieldArray(tools::Tools, n::Int)
    @unpack N, d = tools.sys
    dims = vcat([n], [N for _ in 1:d])
    fx = zeros(dims...)
    return FieldArray(tools, fx, n)
end

"""
field_names is a list of either symbols like :φ, where φ is the name of the field,
or a tuple (:φ, n), where n is the numbers of components of the field φ.
Returns a NamedTuple where "field" such that field[:φ] is a Field or a FieldList.
"""
# TODO: Is there a better way to create NamedTuple?
function get_fields(tools, field_names)
    fields = Dict{Symbol, Fields}()
    for name in field_names
        if typeof(name)==Symbol
            push!(fields, name=>Field(tools))
        else
            push!(fields, name[1]=>FieldArray(tools, name[2]))
        end
    end
    return NamedTuple(fields)
end

