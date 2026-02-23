export get_mesh_name
export output_directory


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