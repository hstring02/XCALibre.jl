using Plots
using XCALibre
# using CUDA


grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "cylinder_d10mm_5mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

mesh_dev = mesh
# mesh_dev = adapt(CUDABackend(), mesh)

# INLET CONDITIONS 

Umag = 2.5
velocity = [Umag, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-5
νR = 5
Tu = 0.01
k_inlet = 3/2*(Tu*Umag)^2
ω_inlet = k_inlet/(νR*nu)
Re = (0.2*velocity[1])/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    # turbulence = RANS{KOmega}(β⁺=0.09),
    turbulence = RANS{KOmega}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Neumann(:top, 0.0),
    Neumann(:bottom, 0.0),
    # Dirichlet(:top, velocity),
    # Dirichlet(:bottom, velocity),
    Wall(:cylinder, noSlip)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:top, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:cylinder, 0.0)
)

@assign! model turbulence k (
    Dirichlet(:inlet, k_inlet),
    Neumann(:outlet, 0.0),
    Neumann(:top, 0.0),
    Neumann(:bottom, 0.0),
    Dirichlet(:cylinder, 1e-15)
)

@assign! model turbulence omega (
    Dirichlet(:inlet, ω_inlet),
    Neumann(:outlet, 0.0),
    Neumann(:top, 0.0),
    Neumann(:bottom, 0.0),
    OmegaWallFunction(:cylinder) # need constructor to force keywords
)

@assign! model turbulence nut (
    Dirichlet(:inlet, k_inlet/ω_inlet),
    Neumann(:outlet, 0.0),
    Neumann(:top, 0.0),
    Neumann(:bottom, 0.0), 
    Dirichlet(:cylinder, 0.0)
)

schemes = (
    U = set_schemes(divergence=Upwind, gradient=Midpoint, limiter=MFaceBased(model.domain)),
    p = set_schemes(gradient=Midpoint),
    k = set_schemes(divergence=Upwind, gradient=Midpoint),
    omega = set_schemes(divergence=Upwind, gradient=Midpoint)

    # U = set_schemes(divergence=Upwind),
    # p = set_schemes(divergence=Upwind),
    # k = set_schemes(divergence=Upwind),
    # omega = set_schemes(divergence=Upwind)
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.3,
    ),
    k = set_solver(
        # model.turbulence.fields.k;
        model.turbulence.k;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
    ),
    omega = set_solver(
        # model.turbulence.fields.omega;
        model.turbulence.omega;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver, CgSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
    )
)

runtime = set_runtime(iterations=1000, write_interval=100, time_step=1)

# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
hardware = set_hardware(backend=CPU(), workgroup=1024)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, k_inlet/ω_inlet)

residuals = run!(model, config, ncorrectors=1); #, pref=0.0)

Reff = stress_tensor(model.momentum.U, nu, model.turbulence.nut)
Fp = pressure_force(:cylinder, model.momentum.p, 1.25)
Fv = viscous_force(:cylinder, model.momentum.U, 1.25, nu, model.turbulence.nut)

plot(; xlims=(0,runtime.iterations), ylims=(1e-10,0))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")