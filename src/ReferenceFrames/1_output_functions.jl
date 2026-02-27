export save_output_polar



function save_output_polar(model::Physics{T,F,SO,M,Tu,E,D,BI}, outputWriter, iteration, time, config, x0, rotaxis
    ) where {T,F,SO,M,Tu,E,D,BI}
    U = model.momentum.U
    mesh = U.mesh
    cells = mesh.cells
    Up = VectorField(mesh)

    for i ∈ eachindex(Up.x.values)
        r = cells[i].centre - x0
        r_norm = r./norm(r)
        tang = r_norm × rotaxis
        Up.x.values[i] = U[i] ⋅ r_norm
        Up.z.values[i] = U.z.values[i]
        Up.y.values[i] = U[i] ⋅ tang
    end
    args = (
        ("U", model.momentum.U), 
        ("Up", Up), 
        ("p", model.momentum.p)
    )
    write_results(iteration, time, model.domain, outputWriter, config.boundaries, args...)
end