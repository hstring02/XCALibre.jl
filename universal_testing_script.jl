using Plots
using XCALibre


# mesh_file = (raw"C:\Users\Harry\OneDrive - The University of Nottingham\1 - Documents\Year 5\01_MEng_Project\3_FlowSims\04_MRF_vs_SRF_vs_norm\01_Meshing\SpinningCylinder0p2Diameter\SpinningCylinder0p2Diameter.unv")
mesh_file = (raw"C:\Users\Harry\OneDrive - The University of Nottingham\1 - Documents\Year 5\01_MEng_Project\3_FlowSims\04_MRF_vs_SRF_vs_norm\01_Meshing\SpinningSquare0p2Diameter\SpinningSquare0p2Diameter.unv")
# mesh_file = (raw"C:\Users\Harry\OneDrive - The University of Nottingham\1 - Documents\Year 5\01_MEng_Project\3_FlowSims\04_MRF_vs_SRF_vs_norm\01_Meshing\SpinningPlate0p3Diameter\SpinningPlate0p3Diameter.unv")
mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU(); workgroup = 1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

nu = 1e-3
velocity = [0.0, 0.0, 0.0]
Tu = 0.05
nuR = 100
k_inlet = 1 #3/2*(Tu*u_mag)^2
ω_inlet = 1000 #k_inlet/(nuR*nu)
νt_inlet = k_inlet/ω_inlet
Re = velocity[1]*0.1/nu

# Rotating reference frames
omega = 6.0
rotaxis = SVector([0.0, 0.0, 1.0]){3}
x0 = SVector([0.0, 0.0, 0.0]){3}
# zones = ScalarField(mesh).+1
# MRF = MRF(omega, rotaxis, x0, zones)
SRF = SRF(omega, rotaxis, x0)

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KOmega}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev,
    REF_FRAME = SRF
    )

BCs = assign(
    region = mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Extrapolated(:outlet),
            Wall(:wall, [0.0, 0.0, 0.0]),
            Wall(:top, [0.0, 0.0, 0.0])
        ],
        p = [
            Neumann(:inlet, 0.0),
            Dirichlet(:outlet, 0.0),
            Wall(:wall),
            Wall(:top)
        ],
        k = [
            Dirichlet(:inlet, k_inlet),
            Extrapolated(:outlet),
            KWallFunction(:wall),
            KWallFunction(:top)
        ],
        omega = [
            Dirichlet(:inlet, ω_inlet),
            Extrapolated(:outlet),
            OmegaWallFunction(:wall),
            OmegaWallFunction(:top)
        ],
        nut = [
            Dirichlet(:inlet, νt_inlet),
            Extrapolated(:outlet),
            NutWallFunction(:wall), 
            NutWallFunction(:top)
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


# Custom Output Functions
mesh_name = get_mesh_name(mesh_file)
velocity_name = string("velocity_",u_mag)*'_'
omega_name = string("omega_",omega)*'_'
script_name = string(@__FILE__)
output_dir = mesh_name * omega_name * "_polarCoords"
pattern = "vtk"
# pattern = "foam"
output_directory(output_dir, script_name)
