export Laminar

# Model type definition (hold fields)
"""
    Laminar <: AbstractTurbulenceModel

Laminar model definition for physics API.
"""
struct Laminar <: AbstractRANSModel end 
Adapt.@adapt_structure Laminar

# Model type definition (hold equation definitions and internal data)
struct LaminarModel{S1}
    state::S1 # required field for all turbulence models
end 
Adapt.@adapt_structure LaminarModel

# Model API constructor (pass user input as keyword arguments and process if needed)
RANS{Laminar}() = begin # Empty constructor
    args = (); ARG = typeof(args)
    RANS{Laminar,ARG}(args)
end

# Functor as constructor (internally called by Physics API): Returns fields and user data
(rans::RANS{Laminar, ARG})(mesh) where ARG = Laminar()

# Model initialisation
"""
    function initialise(
        turbulence::Laminar, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn
        ) where {T,F,M,Tu,E,D,BI}
    return LaminarModel()
end

Initialisation of turbulent transport equations.

### Input
- `turbulence` -- turbulence model.
- `model`  -- Physics model defined by user.
- `mdtof`  -- Face mass flow.
- `peqn`   -- Pressure equation.

### Output
- `LaminarModel()`  -- Turbulence model structure.

"""
function initialise(
    turbulence::Laminar, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn
    ) where {T,F,M,Tu,E,D,BI}
    state = ModelState((), true) # stores residual and convergence information
    return LaminarModel(state)
end

# Model solver call (implementation)
"""
    turbulence!(rans::LaminarModel, model::Physics{T,F,M,Tu,E,D,BI}, S, prev, time
    ) where {T,F,M,Tu<:Laminar,E,D,BI}

Run turbulence model transport equations.

### Input
- `rans::LaminarModel` -- Laminar turbulence model.
- `model`  -- Physics model defined by user.
- `S`   -- Strain rate tensor.
- `prev`  -- Previous field.
- `time`   -- 

"""
function turbulence!(rans::LaminarModel, model::Physics{T,F,M,Tu,E,D,BI}, S, prev, time
    ) where {T,F,M,Tu<:AbstractTurbulenceModel,E,D,BI}
    nothing
end

function turbulence!(
    rans::LaminarModel, model::Physics{T,F,M,Tu,E,D,BI}, S, prev, time
    ) where {T,F<:AbstractCompressible,M,Tu<:AbstractTurbulenceModel,E,D,BI}
    config = get_configuration(CONFIG)
    (; U, Uf, gradU) = S
    grad!(gradU, Uf, U, config.boundaries.U, time)
    limit_gradient!(config.schemes.U.limiter, gradU, U)
    nothing
end

# Specialise VTK writer
function save_output(model::Physics{T,F,M,Tu,E,D,BI}, outputWriter, iteration, time
    ) where {T,F,M,Tu<:Laminar,E,D,BI}
    if typeof(model.fluid)<:AbstractCompressible
        args = (
            ("U", model.momentum.U), 
            ("p", model.momentum.p),
            ("T", model.energy.T)
        )
    else
        args = (
            ("U", model.momentum.U), 
            ("p", model.momentum.p)
        )
    end
    config = get_configuration(CONFIG)
    write_results(iteration, time, model.domain, outputWriter, config.boundaries, args...)
end

function save_output(model::Physics{T,F,M,Tu,E,D,BI}, outputWriter, iteration, time
    ) where {T,F,M,Tu<:Laminar,E<:Nothing,D,BI}
    args = (
        ("U", model.momentum.U), 
        ("p", model.momentum.p),
    )
    config = get_configuration(CONFIG)
    write_results(iteration, time, model.domain, outputWriter, config.boundaries, args...)
end