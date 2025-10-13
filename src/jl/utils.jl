include("structs.jl")
using Statistics

################################
# Utilities for saving to disk #
################################

function get_itr_names(save_names)
    # Field name, component number, number in list of all fields to save
    itr_name = []
    for name in save_names
        if typeof(name)==Symbol; push!(itr_name, (name, nothing))
        else for i in 1:name[2]; push!(itr_name, (name[1], i)) end
        end
    end
    return itr_name
end

function save_field(fields, save_opt, m,  mode)
    @unpack save_names, folder = save_opt

    itr_name = get_itr_names(save_names)
    for (name, i) in itr_name
        file_name = folder * string(name)
        if !isnothing(i) file_name *= "_"*string(i) end
        
        data_name = (@sprintf ("step_%05d") m)
        @assert !any(isnan.(fields[name][i].x))
        @assert !any(isinf.(fields[name][i].x))

        jldopen(file_name, mode) do file file[data_name] = fields[name][i].x end 
    end
    nothing
end


function average_q!(f, fq, a, abs_p, p, ::Tools{1})
    for i in eachindex(fq[a, :]) if (abs(f[i]) > 1e-20) fq[a, i] = f[i] end end
end

function average_q!(f, fq, a, abs_p, p, ::Tools{2})
    samples = zero(p)
    for j in eachindex(f)
        # find i so that p[indx[j]-1] < abs_p[j] <= p[indx[j]]
        i = searchsortedlast(p, abs_p[j])
        if (abs(f[j]) > 1e-20)
            fq[a,i] += f[j]
            samples[i] += 1
        end
    end
    mt = (samples .== 0) # Empty
    @. fq[a,!mt] = fq[a,!mt] / samples[!mt]
end


function save_corr(fields, save_opt, m, tools, mode)
    @unpack save_names, folder = save_opt
    @unpack L, N, d = tools.sys
    
    itr_name = get_itr_names(save_names)
    n_fields = length(itr_name)
    if d==1 fpf = tools.fplan
    else fpf = tools.fplan_full end
    name, i = itr_name[1]
    f1 = fpf * fields[name][i].x

    p = tools.p[1]
    if d==1 abs_p = p
    else abs_p = @. sqrt((tools.p[2]^2) + (tools.p[2]^2)') end

    C       = zeros(typeof(f1[1]), (size(f1)...) )
    Cabq    = zeros(typeof(f1[1]), (n_fields, n_fields, size(p)...) )
    faq     = zeros(typeof(f1[1]), (n_fields, size(p)...) )

    for (name_a, a) in itr_name
        fa = ( fpf * fields[name_a][a].x) * (L/N)^d
        average_q!(fa, faq, a, abs_p, p, tools)

        for (name_b, b) in itr_name
            fb = ( fpf * fields[name_b][b].x) * (L/N)^d
            @. C = fa * conj(fb) * (1/L)^d
            ab = CartesianIndex(a,b)
            average_q!(C, Cabq, ab, abs_p, p, tools)
        end
    end

    data_name = (@sprintf ("step_%05d") m)
    jldopen(folder * "φaq", mode) do file file[data_name] = faq end 
    jldopen(folder * "Cabq", mode) do file file[data_name] = Cabq end 
end


function save_first(tools, con, fields, save_opt)
    @unpack folder, N_step, t_start = save_opt
    mode = "w"

    if ! isdir(folder) mkpath(folder) end
    jldopen(folder*"parameters", "w") do file
        file["con"] = con
        if :tools in save_opt.SAVEDATA file["tools"] = tools end
    end 

    data_name = (@sprintf ("TIME_%05d") 0)
    jldopen(folder*"TIME", "w") do file file[data_name] = 0 end

    if save_opt.SAVEFIELD save_field(fields, save_opt, 0, mode) end
    if save_opt.SAVECORR save_corr(fields, save_opt, 0, tools, mode) end
    if save_opt.N_write>0 write_stat(save_opt, 0) end

    if :SAVEDATA in keys(save_opt)
        for key in save_opt.SAVEDATA
            if key==:tools nothing
            else save_data_fns[key](fields, save_opt, 0, tools, mode) end
        end
    end
    nothing
end


function check_and_save(fields, tools, i, save_opt)
    @unpack N_step, N_save, N_write, folder = save_opt
    int = N_step ÷ N_save  # steps between saves
    mode = "a+"
    if i % int == 0
        @unpack Δt = tools.sys
        TIME = i * Δt
        m = i ÷ int
        data_name = (@sprintf ("TIME_%05d") m)
        jldopen(folder*"TIME", mode) do file file[data_name] = TIME end

        if save_opt.SAVEFIELD save_field(fields, save_opt, m, mode) end
        if save_opt.SAVECORR save_corr(fields, save_opt, m, tools, mode) end

        if :SAVEDATA in keys(save_opt)
            for key in save_opt.SAVEDATA
                if key==:tools nothing
                else save_data_fns[key](fields, save_opt, m, tools, mode) end
            end
        end
    end
    if  (N_write>0) && ((i%(N_step÷N_write))==0)  write_stat(save_opt, i) end
    nothing
end


function save_info(numbers, folder)
    if ! isdir(folder) mkpath(folder) end
    jldopen(folder*"info", "w") do file
        file["numbers"] = numbers
    end
end


function write_stat(save_opt, i)
    @unpack N_step, t_start = save_opt
    format = "dd-mm-yyyy \nHH:MM:SS"
    progress = i / N_step
    
    stat = "Status simulation:\n\n"
    first = "Simulation started:\n" * Dates.format(t_start, format)*"\n\n"
    last = "Last updated:\n" * Dates.format(now(), format)*"\n\n"
    perc = round(progress*100; digits=1)
    bars = 25
    perd = round(Int, progress*bars)
    bar = "Progress: $(perc)%\n[" * rpad("|"^perd, bars) * "]\n\n"
    dt = now() - t_start
    run = "Run time:\n" * string(Dates.canonicalize(dt)) * "\n\n"
    tleft = "∞"
    if i>0 
        tleft_f = round(Dates.value(dt)*(1/progress-1), digits=0)
        tleft = string(Dates.canonicalize(Millisecond(tleft_f)))  
    end
    if i==N_step tleft = "0" end
    left = "Expected time left:\n"* tleft  * "\n"

    name = "status.txt"
    if :stat_path in keys(save_opt)
        # This folder must be created manually
        path = save_opt.stat_path*name
    else
        path = save_opt.folder*name
    end

    write(path, stat * first * last * bar * run * left)
end


##########################
# Additional Observalbes #
##########################


function save_J(fields, save_opt, m, tools::Tools{2}, mode)
    #TODO: Implement in 2 dim
    @unpack folder = save_opt
    @unpack p, sys, bplan = tools
    @unpack d = sys
    @unpack φ, ∇φ = fields
    Nf = size(φ)[1]
    for a in 1:Nf
        for x in eachindex(p[1])
            for y in eachindex(p[2])
                r = (x, y)
                for i in 1:d
                    ∇φ[(a-1)*d+1 + i-1].k[x,y] = 1im * p[i][r[i]] * φ[a].k[x,y]
                end
            end
        end
    end

    for ∇φi in ∇φ ∇φi.x = bplan * ∇φi.k end

    J = zero(φ[1].x)
    for i in 1:d
        # Spatial and species index contained in same index
        # i counts over spatial dimension first, then 1, 2, over species
        @. J += (φ[1].x * ∇φ[(2-1)*d+1 + i-1].x - φ[2].x * ∇φ[(1-1)*d+1 + i-1].x)^2
        #     = (φ[1].x * ∇φ[d+i].x - φ[2].x * ∇φ[i].x)^2
    end
    @. J = sqrt(J)
    data_name = (@sprintf ("step_%05d") m)
    file_name = folder * string("J")
    jldopen(file_name, mode) do file file[data_name] = mean(J) end 
    nothing
end

const save_data_fns = (J = save_J, )


############################
# Functions for integrator #
############################

# Fix, for the correlations to be right given real fourier transform
function fix_corr!(ξ::Array{ComplexF64, 1}, tools)
    @unpack N = tools.sys
    @views ξ[1] = real(ξ[1]) * sqrt(2)
    @views ξ[N÷2+1] = real(ξ[N÷2+1]) * sqrt(2)
end

# Fix, for the correlations to be right given real fourier transform
# TODO: genralize to higer dim
function fix_corr!(ξ::Array{ComplexF64, 2}, tools)
    @unpack N = tools.sys
    for i in 2:N÷2
        @views ξ[1, i] = conj(ξ[1, end+2 - i])
        @views ξ[end, i] = conj(ξ[end, end+2 - i])
    end
    
    @views ξ[1, 1] = real(ξ[1, 1]) * sqrt(2)
    @views ξ[1, N÷2+1] = real(ξ[1, N÷2+1]) * sqrt(2)
    @views ξ[N÷2+1, 1] = real(ξ[N÷2+1, 1]) * sqrt(2)
    @views ξ[N÷2+1, N÷2+1] = real(ξ[N÷2+1, N÷2+1]) * sqrt(2)
end

"""
Antialiaz a square in fourier space with sides 2pAA
"""
function AA!(φ, tools)
    @unpack pAA, p, sys = tools
    for i in CartesianIndices(φ.k); for j in 1:sys.d
        if abs(p[j][i[j]])<pAA continue
        else φ.k[i] = 0; break end 
    end end
end


function wall_positions(field, tools)
    @unpack N, L = tools.par
    @unpack x = tools
    @unpack φ = field
    
    zerp = Float64[]
    zerm = Float64[]
    for i in 1:N
        if φ[i] * φ[i%N+1] < 0
            
            x_1, x_2 = x[i], x[i%N+1]
            y_1, y_2 = φ[i], φ[i%N+1]
    
            if (x_2 < x_1) x_2 = x_2 + L end
            
            x0 = x_1 - (x_2 - x_1) * y_1/(y_2 - y_1)
            x0 = x0 % L
            
            if φ[i] < 0 push!(zerm, x0)
            else push!(zerp, 0)
            end
        end
    end
    
    return vcat(zerm, zerp)
end


function init_square!(fields, tools, sys, S, dist)
    @unpack d = sys
    for i in axes(S, 1)
        φ0 = zero(fields[:φ][i].x)
        φ0 .= -1.
        ind = [(dist[i]:dist[i]+S[i]) for _ in 1:d]
        φ0[ind...] .= 1.

        fields[:φ][i].x .= φ0
        fields[:φ][i].k .= tools.fplan * fields[:φ][i].x
    end
end 

function init_hom!(fields, tools, sys, bget_itr_namesφ)
    @unpack d = sys
    for i in axes(bφ, 1)
        fields[:φ][i].x .= bφ[i]
        fields[:φ][i].k .= tools.fplan * fields[:φ][i].x
    end
end 

function init2fields!(fields, tools, con)
    @unpack x, seed = tools
    @unpack L, N, d = tools.sys
    rng = MersenneTwister(seed)

    dims = Tuple(N for _ in 1:d)
    f1 = rand(rng, Float32, dims) .* 1.
    f2 = rand(rng, Float32, dims) .* 1.
    f1 = f1 .- mean(f1) .+ con[:bφ]
    f2 = f2 .- mean(f2)
    @. fields.φ[1].x = f1
    @. fields.φ[2].x = f2
    fields.φ[1].k = tools.fplan * fields.φ[1].x
    fields.φ[2].k = tools.fplan * fields.φ[2].x
end

