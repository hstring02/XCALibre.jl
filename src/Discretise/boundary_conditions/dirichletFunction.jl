export XCALibreUserFunctor
export DirichletFunction

abstract type XCALibreUserFunctor end

"""
    DirichletFunction(ID, value) <: AbstractDirichlet

Dirichlet boundary condition defined with user-provided function.

# Input
- `ID` Name of the boundary given as a symbol (e.g. :inlet). Internally it gets replaced with the boundary index ID
- `value` Custom function or struct (<:XCALibreUserFunctor) for Dirichlet boundary condition
- `IDs_range` Range of indices to access boundary patch faces

# Function requirements

The function passed to this boundary condition must have the following signature:

    f(coords, time, index) = SVector{3}(ux, uy, uz)

Where, `coords` is a vector containing the coordinates of a face, `time` is the current time in transient simulations (and the iteration number in steady simulations), and `index` is the local face index (from 1 to `N`, where `N` is the number of faces in a given boundary). The function must return an SVector (from StaticArrays.jl) representing the velocity vector. 
"""
struct DirichletFunction{I,V,R<:UnitRange} <: AbstractDirichlet
    ID::I 
    value::V
    IDs_range::R
end
Adapt.@adapt_structure DirichletFunction

adapt_value(value::XCALibreUserFunctor, mesh) = begin
    value
end

@define_boundary DirichletFunction Laplacian{Linear} begin
    J = term.flux[fID]
    (; area, delta, centre) = face 
    flux = J*area/delta
    ap = (-flux)
    # bc.value.update!(bc.value, centre, time, i)
    value = bc.value(centre, time, i)[component.value]
    ap, ap*value
end

@define_boundary DirichletFunction Divergence{Linear} begin
    flux = -term.flux[fID]
    ap = (flux)
    value = bc.value(face.centre, time, i)[component.value]
    0.0, ap*value
end

@define_boundary DirichletFunction Divergence{Upwind} begin
    flux = -term.flux[fID]
    ap = (flux)
    value = bc.value(face.centre, time, i)[component.value]
    0.0, ap*value
end

@define_boundary DirichletFunction Divergence{BoundedUpwind} begin
    flux = -term.flux[fID]
    ap = (flux)
    value = bc.value(face.centre, time, i)[component.value]
    flux, ap*value
end

@define_boundary DirichletFunction Si begin
    0.0, 0.0
end
