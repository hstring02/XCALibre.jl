using XCALibre
using LinearAlgebra

mesh_file = (raw"C:\Users\Harry\OneDrive - The University of Nottingham\1 - Documents\Year 5\01_MEng_Project\3_FlowSims\03_SRF_Testing\01_Meshing\Square_In_Circle_V3_100mm.unv")
mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU(); workgroup = 1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

nu = 1e-3
u_mag = 0.0
velocity = [u_mag, 0.0, 0.0]
Tu = 0.05
nuR = 100
k_inlet = 1 #3/2*(Tu*u_mag)^2
ω_inlet = 1000 #k_inlet/(nuR*nu)
νt_inlet = k_inlet/ω_inlet
Re = velocity[1]*0.1/nu

# SRF
omega = 5.0
x0 = [0.0, 0.0, 0.0] # Centre of rotation
x1 = [0.0, 0.0, 3.0] # Arbitrary point along rotation axis
S = (x1 - x0)./norm((x1 - x0))

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KOmega}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev,
    omega = omega,
    S = S,
    x0 = x0
    )

BCs = assign(
    region = mesh_dev,
    (
        U = [
            RotatingWall(:Boundary, rpm=omega*(30/pi), centre=x0, axis=S),
            Wall(:Walls, [0.0, 0.0, 0.0])
        ],
        p = [
            # Neumann(:Boundary, 0.0),
            Dirichlet(:Boundary, 0.0),
            Wall(:Walls)
        ],
        k = [
            Dirichlet(:Boundary, k_inlet),
            KWallFunction(:Walls)
        ],
        omega = [
            Dirichlet(:Boundary, ω_inlet),
            OmegaWallFunction(:Walls)
        ],
        nut = [
            Dirichlet(:Boundary, νt_inlet),
            NutWallFunction(:Walls)
        ]
    )
)

schemes = (
    U = Schemes(divergence=Upwind),
    p = Schemes(divergence=Upwind),
    k = Schemes(divergence=Upwind),
    omega = Schemes(divergence=Upwind)
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-2,
        atol = 1e-10
    ),
    p = SolverSetup(
        solver      = Cg(), #Gmres(), #Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-3,
        atol = 1e-10
    ),
    k = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-2,
        atol = 1e-10
    ),
    omega = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-2,
        atol = 1e-10
    )
)

runtime = Runtime(iterations=3000, write_interval=300, time_step=1)
# runtime = Runtime(iterations=2, write_interval=-1, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)


GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, νt_inlet)

residuals = run!(model, config) # 145 iterations