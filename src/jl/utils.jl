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

function save_corr(fields, save_opt, m, tools::Tools{1}, mode)
    @unpack save_names, folder = save_opt
    itr_name = get_itr_names(save_names)
    n_fields = length(itr_name)

    fpf = tools.fplan
    @unpack L, N, d = tools.sys

    name, i = itr_name[1]
    f1 = fpf * fields[name][i].x
    p = tools.p[1]

    C = zeros(typeof(f1[1]), (n_fields, n_fields, size(p)...) )

    for (name_a, a) in itr_name
        for (name_b, b) in itr_name
            fa = ( fpf * fields[name_a][a].x)
            fb = ( fpf * fields[name_b][b].x)

            if n_fields==1 @. C = fa * conj(fb) * (L/N^2)^d
            else @. C[a,b,:] = fa * conj(fb) * (L/N^2)^d end
        end
    end

    data_name = (@sprintf ("step_%05d") m)
    file_name = folder * "Cab"
    jldopen(file_name, mode) do file file[data_name] = C end 
end


function save_corr(fields, save_opt, m, tools::Tools{2}, mode)
    #TODO: Remove fplan_full use ξ not C_vec, combine n_fields==1, C and f
    @unpack save_names, folder = save_opt
    itr_name = get_itr_names(save_names)
    n_fields = length(itr_name)

    fpf = tools.fplan_full
    @unpack L, N, d = tools.sys

    name, i = itr_name[1]
    f1 = fpf * fields[name][i].x
    
    C_vec = zeros(typeof(f1[1]), (size(f1)...) )

    p       = tools.p[1]
    p_full  = tools.p[2]
    abs_p   = sqrt.((p_full .^2) .+ (p_full .^2)')

    if n_fields==1 
        C = zeros(typeof(f1[1]), size(p)... ) 
        faq = zeros(typeof(f1[1]), size(p)... ) 
    else
        C = zeros(typeof(f1[1]), (n_fields, n_fields, size(p)...) ) 
        faq = zeros(typeof(f1[1]), (n_fields, size(p)...) ) 
    end


    for (name_a, a) in itr_name
        fa = ( fpf * fields[name_a][a].x) * (L/N)^d
        samples = zero(p)
        for j in eachindex(fa)
            i = searchsortedlast(p, abs_p[j])
            if (abs(fa[j]) > 1e-10)
                if n_fields==1 faq[i] += fa[j]
                else faq[a,i] += fa[j] end
                samples[i] += 1
            end
        end
        mt = (samples .== 0) # Empty
        if n_fields==1 @. faq[!mt] = faq[!mt] / samples[!mt]
        else @. faq[a,!mt] = faq[a,!mt] / samples[!mt] end


        for (name_b, b) in itr_name
            fb = ( fpf * fields[name_b][b].x)
            @. C_vec = fa * conj(fb) * (L/N^2)^d
            samples = zero(p)
            
            for j in eachindex(C_vec)
                # find i so that p[indx[j]-1] < abs_p[j] <= p[indx[j]]
                i = searchsortedlast(p, abs_p[j])
                if (abs(C_vec[j]) > 1e-20)
                    if n_fields==1 C[i] += C_vec[j]
                    else C[a,b,i] += C_vec[j] end
                    samples[i] += 1
                end
            end
            mt = (samples .== 0) # Empty
            if n_fields==1 @. C[!mt] = C[!mt] / samples[!mt]
            else @. C[a,b,!mt] = C[a,b,!mt] / samples[!mt] end
        end
    end

    data_name = (@sprintf ("step_%05d") m)
    jldopen(folder * "φaq", mode) do file file[data_name] = faq end 
    jldopen(folder * "Cabq", mode) do file file[data_name] = C end 
end


function save_J(fields, save_opt, m, tools, mode)
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
        @. J += (φ[1].x * ∇φ[(2-1)*d+1 + i-1].x - φ[2].x * ∇φ[(1-1)*d+1 + i-1].x)^2
    end
    @. J = sqrt(J)

    data_name = (@sprintf ("step_%05d") m)

    file_name = folder * string("J")

    jldopen(file_name, mode) do file file[data_name] = mean(J) end 

    nothing
end

save_data_fns = (J = save_J, )


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
    if ! isdir(folder)
        mkpath(folder)
    end
    jldopen(folder*"info", "w") do file
        file["numbers"] = numbers
    end
end


function write_stat(save_opt, i)
    @unpack N_step, folder, t_start = save_opt
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
    path =  folder*name

    write(path, stat * first * last * bar * run * left)
end


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
