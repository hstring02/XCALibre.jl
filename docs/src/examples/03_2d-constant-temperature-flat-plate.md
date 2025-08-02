# Validation: 2D Constant temperature flat plate

# Introduction
---
A 2D constant temperature laminar flat plate case has been used to validate the weakly 
compressible solver. The case provides a constant temperature boundary condition along the 
wall of the domain. 

The Nusselt number values obtained from the simulation are compared against the theoretical local Nusselt number correlation 
for forced convection on constant temperature flat plate:

``Nu_x = 0.332 Re_x^{1/2} Pr^{1/3}``

The correlation is valid for Prandtl numbers greater than 0.6.

## Boundary conditions

### Inlet


| Field | Boundary condition      |
| -------  | ---------- |
| ``U``   |  Dirichlet ([0.2, 0.0, 0.0] m/s)   |
| ``p``   |  Extrapolated (Zero-gradient)      |
| ``T ``  |  FixedTemperature (300.0 K)    |

### Wall


| Field | Boundary condition      |
| -------  | ---------- |
| ``U``   |  No-slip wall   |
| ``p``   |  Extrapolated (Zero-gradient)      |
| ``T ``  |  FixedTemperature (310.0 K)    |

### Outlet


| Field | Boundary condition      |
| -------  | ---------- |
| ``U``   |  Extrapolated (Zero-gradient)   |
| ``p``   |  Dirichlet (0.0 Pa)     |
| ``T ``  |  Extrapolated (Zero-gradient)    |

### Top


| Field | Boundary condition      |
| -------  | ---------- |
| ``U``   |  Extrapolated (Zero-gradient)   |
| ``p``   |  Extrapolated (Zero-gradient)     |
| ``T ``  |  Extrapolated (Zero-gradient)    |


# Fluid Properties


| Property | value      |
| -------  | ---------- |
| ``\nu``   |  0.0001    |
| ``Pr``   |  0.71      |
| ``c_p``  |  1005.0    |
| ``\gamma`` | 1.4      |



# Mesh
---

The mesh is shown in the figure below. The plate is represented by the "wall" boundary. The flow moves from left to right ("inlet" to "outlet" boundaries). The x-axis is aligned with the wall boundary and the y-axis is position in the direction perpendicular to the wall.


![Figure 1](figures/03/mesh.png)

The streamwise cell length is 2mm with a total domain length of 1m. The near-wall cell height is 0.049mm with a domain height of 0.2m.


# Case file
---

```jldoctest;  filter = r".*"s => s"", output = false
using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "flatplate_2D_laminar.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

velocity = [0.2, 0.0, 0.0]
nu = 1e-4
Re = velocity[1]*1/nu
cp = 1005.0
gamma = 1.4
Pr = 0.7

model = Physics(
    time = Steady(),
    fluid =  Fluid{WeaklyCompressible}(
        nu = nu,
        cp = cp,
        gamma = gamma,
        Pr = Pr
        ),
    turbulence = RANS{Laminar}(),
    energy = Energy{SensibleEnthalpy}(Tref = 288.15),
    domain = mesh # mesh_dev  # use mesh_dev for GPU backend
    )

BCs = assign(
    region=mesh,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Extrapolated(:outlet),
            Wall(:wall, [0.0, 0.0, 0.0]),
            Symmetry(:top)
        ],
        p = [
            Extrapolated(:inlet),
            Dirichlet(:outlet, 100000.0),
            Wall(:wall),
            Symmetry(:top)
        ],
        h = [
            FixedTemperature(:inlet, T=300.0, Enthalpy(cp=cp, Tref=288.15)),
            Extrapolated(:outlet),
            FixedTemperature(:wall, T=310.0, Enthalpy(cp=cp, Tref=288.15)),
            Symmetry(:top)
        ],
    )
)

schemes = (
    U = Schemes(divergence=Linear),
    p = Schemes(divergence=Linear),
    h = Schemes(divergence=Linear)
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-1
    ),
    p = SolverSetup(
        solver      = Cg(),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-1
    ),
    h = SolverSetup(
        solver      = Bicgstab(),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-1
    )
)

runtime = Runtime(iterations=1000, write_interval=1000, time_step=1)
runtime = Runtime(iterations=1, write_interval=-1, time_step=1) # hide

hardware = Hardware(backend=CPU(), workgroup=4)

configure!(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 100000.0)
initialise!(model.energy.T, 300.0)

residuals = run!(model)

# output

```

# Results
---

The results of the model are compared to the theoretical correlation in the figure below:

![Nusselt number distribution results.](figures/03/Nusselt_const_temp_lam_plate.png)
