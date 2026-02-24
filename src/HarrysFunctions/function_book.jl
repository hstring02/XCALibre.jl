export get_mesh_name
export output_directory
export make_absolute_velocity!
export _make_absolute_velocity!
export update_srf_sources!
export _update_srf_sources!
# export update_mrf_sources!
# export _update_mrf_sources!

# These are all of hte functions that I have written for use across both the 
# SRF and MRF implementation, as well as a few quality of life additions.


# Output writing ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

function get_mesh_name(s::String)
    parts = split(s, '\\')
    part = parts[end]
    mesh = split(part, '.')
    return mesh[1]*"__"
end

function output_directory(output_dir::String, script_name::String, pattern::String="vtk", overwrite::Bool=true)
    # Make output directory
    if !isdir(output_dir)
        mkdir(output_dir)
    end

    # Find and move CFD outputs
    for file in readdir()
        if isfile(file) && occursin(pattern, file)
            mv(file, joinpath(output_dir, file); force=true)
        end
    end

    # Save a copy of input script for traceability
    source = script_name
    base = splitext(basename(source))[1]   # filename without extension
    dest = joinpath(output_dir, base * ".txt")
    cp(source, dest; force=overwrite)
end


# SRF functions ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

function update_srf_sources!(omegaU, omegaR, U, x0, rotaxis, omega, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    mesh = U.mesh
    cells = mesh.cells 

    ndrange = length(cells)
    kernel! = _update_srf_sources!(_setup(backend, workgroup, ndrange)...)
    kernel!(omegaU, omegaR, U, x0, rotaxis, omega, cells)
end

@kernel function _update_srf_sources!(omegaU, omegaR, U, x0, rotaxis, omega, cells)
    cID = @index(Global)

    Omega = omega*rotaxis
    r = cells[cID].centre - x0
    omegaR[cID] = Omega × Omega × r
    omegaU[cID] = 2*Omega × U[cID]
end

function make_absolute_velocity!(U,  x0, rotaxis, omega, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    mesh = U.mesh
    cells = mesh.cells 

    ndrange = length(cells)
    kernel! = _make_absolute_velocity!(_setup(backend, workgroup, ndrange)...)
    kernel!(U,  x0, rotaxis, omega, cells)
    println("Made velocity absolute")

end

@kernel function _make_absolute_velocity!(U,  x0, rotaxis, omega, cells)
    cID = @index(Global)

    Omega = omega*rotaxis
    r = cells[cID].centre - x0
    U[cID] = U[cID] + Omega × r
end


