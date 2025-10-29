using Plots
using XCALibre
using StaticArrays

grids_dir = ("D:/Julia_CFD")
grid = "2D_Cylinder_In_Pipe.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

u_mag = 3
nu = 1e-3
velocity = [u_mag, 0.0, 0.0]
Tu = 0.05
nuR = 100
k_inlet = 1 #3/2*(Tu*u_mag)^2
ω_inlet = 1000 #k_inlet/(nuR*nu)
νt_inlet = k_inlet/ω_inlet
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KOmega}(),
    #turbulence = RANS{KOmegaSST}(walls=(:Sphere,)),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

BCs = assign(
    region = mesh_dev,
    (
        U = [
            Dirichlet(:Inlet, velocity),
            Extrapolated(:Outlet),
            Extrapolated(:Top),
            Extrapolated(:Bottom),
            Wall(:Sphere, [0.0, 0.0, 0.0])
        ],
        p = [
            Neumann(:Inlet, 0.0),
            Dirichlet(:Outlet, 0.0),
            Extrapolated(:Top),
            Extrapolated(:Bottom),
            Wall(:Sphere)
        ],
        k = [
            Dirichlet(:Inlet, k_inlet),
            Extrapolated(:Outlet),
            Extrapolated(:Bottom),
            Extrapolated(:Top),
            KWallFunction(:Sphere)
        ],
        omega = [
            Dirichlet(:Inlet, ω_inlet),
            Extrapolated(:Outlet),
            Extrapolated(:Top),
            Extrapolated(:Bottom),
            OmegaWallFunction(:Sphere)
        ],
        nut = [
            Dirichlet(:Inlet, νt_inlet),
            Extrapolated(:Outlet),
            Extrapolated(:Top),
            Extrapolated(:Bottom),
            NutWallFunction(:Sphere)
        ]
    )
)

schemes = (
    U = Schemes(divergence=Upwind),
    p = Schemes(divergence=Upwind),
    k = Schemes(divergence=Upwind),
    omega = Schemes(divergence=Upwind),
    y = Schemes(gradient=Midpoint)
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
    ),
    y = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-8,
        rtol = 1e-2,
        relax       = 0.9,
    )
)

runtime = Runtime(iterations=3000, write_interval=100, time_step=1)
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

Reff = stress_tensor(model.momentum.U, nu, model.turbulence.nut)
Fp = pressure_force(:wall, model.momentum.p, 1.25)
Fv = viscous_force(:wall, model.momentum.U, 1.25, nu, model.turbulence.nut)

println("Pressure forces = "+str(Fp))
println("Viscous forces = "+str(Fv))


plot(; xlims=(0,494))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")
