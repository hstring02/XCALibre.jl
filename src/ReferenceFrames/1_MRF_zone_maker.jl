export radial_mask!

function radial_mask!(x0, radius, mask, hardware, mesh)
    (; backend, workgroup) = hardware
    cells = mesh.cells 

    ndrange = length(cells)
    kernel! = _radial_mask!(_setup(backend, workgroup, ndrange)...)
    kernel!(x0, radius, mask, cells)
end

@kernel function _radial_mask!(x0, radius, mask, cells)
    cID = @index(Global)

    r = cells[cID].centre - x0
    length = norm(r)
    if length <= radius
        mask[cID] = 1
    end

end



