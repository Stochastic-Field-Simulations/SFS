include("utils.jl")


##############
# Time steps #
##############

function take_step(step::ETD1, fields::T1, tools::Tools; i::Union{Nothing,Int}=nothing) where {T1}
    @unpack K1, K2, σ = step
    @unpack φ, f, ξ = fields
    @unpack bplan, rng = tools
    
    @. φ[i].k = K1 * φ[i].k + K2 * f[i].k + σ * ξ[i].k
    
    AA!(φ[i], tools)
    copyto!(f[i].k, φ[i].k) # Must do this, as mul! modifies 3rd argument...
    mul!(φ[i].x, bplan, f[i].k)
    
    nothing
end


function take_step(step::ETD2O, fields::T1, tools::Tools; i::Union{Nothing,Int}=nothing) where {T1}
    @unpack K1, K2, σ = step
    @unpack φ, f, ξ = fields
    @unpack bplan, rng = tools
    
    @. φ[i].k = K1[i] * φ[i].k + K2[i] * f[i].k + σ[i] * ξ[i].k
    
    AA!(φ[i], tools)
    copyto!(f[i].k, φ[i].k) # Must do this, as mul! modifies 3rd argument...
    mul!(φ[i].x, bplan, f[i].k)
    
    nothing
end


function take_step(step::ETD2, fields::T1, tools::T2; i::Union{Nothing,Int}=nothing) where {T1, T2}
    @unpack K1, K2, K3, σ = step
    @unpack φ, f, f3, ξ = fields
    @unpack bplan, rng = tools
    
    @. φ[i].k = K1 * φ[i].k + K2 * f[i].k + K3 * f3[i].k + σ * ξ[i].k
    copyto!(f3[i].k, f[i].k)
    
    AA!(φ[i], tools)
    copyto!(f[i].k, φ[i].k) # Must do this, as mul! modifies 3rd argument...
    mul!(φ[i].x, bplan, f[i].k)
    
    nothing
end


###############
# Integrators #
###############

# Simple exponential time stepping
function ETD!(fields::T1, tools::Tools, non_lin!::T2, con::T3, i::Int) where {T1, T2, T3}
    @unpack φ, f, ξ = fields
    @unpack fplan, bplan, rng, p2, step = tools

    if i==1 
        if first_step!(step, fields, tools, f!, con); return nothing    # If we take first step, we skip the rest
        else nothing end                                                # If not, we proceed
    end

    non_lin!(fields, con, tools)
    mul!(f.k, fplan, f.x)
    if φ.conserved @. f.k = p2 * f.k end

    randn!(rng, ξ.k)
    fix_corr!(ξ.k, tools)
    take_step(step, fields, tools; i=nothing)

    nothing
end


# Simple exponential time stepping for N fields
function ETDN!(fields::T1, tools::Tools, non_lin!::T2, con::T3, i::Int) where {T1, T2, T3}
    @unpack φ, f, ξ = fields
    @unpack fplan, bplan, rng, p2, step = tools

    if i==1 
        if first_stepN!(step, fields, tools, f!, con); return nothing   # If we take first step, we skip the rest
        else nothing end                                                # If not, we proceed
    end

    non_lin!(fields, con, tools)

    for i in axes(φ, 1)
        mul!(f[i].k, fplan, f[i].x)
        if φ[i].conserved @. f[i].k = p2 * f[i].k end

        randn!(rng, ξ[i].k)
        fix_corr!(ξ[i].k, tools)
        take_step(step, fields, tools; i=i)
    end

    nothing
end


"""
Take first step using first order time-stepping,
save non-linearity to f3 for use in second-order time step
"""

function first_step!(step::ETD2, fields::T1, tools::Tools, non_lin!::T2, con::T3) where {T1, T2, T3}
    # Create new tools for fist order step
    @unpack conserved, sys, seed = tools
    @unpack N, L, T, d, Δt = sys
    @unpack K1, K2, σ = step
    # K2 is not strictly the same as for ETD1, but the same to O(cΔt)²
    step1   = ETD1{d}(K1, K2, σ)
    m       = 10 # factor for first order time step
    sys     = System(d, N, L, Δt/m; T=T)
    tools   = Tools(sys, seed, conserved, step1)

    @unpack φ, f, f3, ξ = fields
    @unpack fplan, bplan, rng, p2, step = tools

    non_lin!(fields, con, tools)
    copyto!(f3.x, f.x)

    for i in 1:m
        non_lin!(fields, con, tools)

        mul!(f.k, fplan, f.x)

        randn!(rng, ξ.k)
        fix_corr!(ξ.k, tools)
        
        if φ.conserved @. f.k = p2 * f.k end
        take_step(step, fields, tools) end
    return true
end


function first_stepN!(step::ETD2, fields::T1, tools::Tools, non_lin!::T2, con::T3) where {T1, T2, T3}
    # Create new tools with fist order step
    @unpack conserved, sys, seed = tools
    @unpack N, L, T, d, Δt = sys
    m       = 10 # factor for first order time step
    sys     = System(d, N, L, Δt/m; T=T)
    @unpack K1, K2, σ = step
    step1 = ETD1{d}(K1, K2, σ)
    tools   = Tools(sys, seed, conserved, step1)

    @unpack φ, f, f3, ξ = fields
    @unpack fplan, bplan, rng, p2, step = tools

    non_lin!(fields, con, tools)
    copyto!(f3, f)

    for i in 1:m
        non_lin!(fields, con, tools)

        for i in axes(φ, 1)
            mul!(f[i].k, fplan, f[i].x)

            randn!(rng, ξ[i].k)
            fix_corr!(ξ[i].k, tools)
            
            if φ[i].conserved @. f[i].k = p2 * f[i].k end
            take_step(step, fields, tools; i=i)
        end
    end
    return true
end


"""
If we use ETD1, we should not take a special first step
"""
function first_step!(::ETD1, ::Any, ::Any, ::Any, ::Any) return false end
function first_step!(::ETD2O, ::Any, ::Any, ::Any, ::Any) return false end
function first_stepN!(::ETD1, ::Any, ::Any, ::Any, ::Any) return false end
function first_stepN!(::ETD2O, ::Any, ::Any, ::Any, ::Any) return false end
