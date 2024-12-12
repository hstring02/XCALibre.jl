export simple!

"""
    simple!(model_in, config; 
        limit_gradient=false, pref=nothing, ncorrectors=0, inner_loops=0)

Incompressible variant of the SIMPLE algorithm to solving coupled momentum and mass conservation equations.

# Input arguments

- `model` reference to a `Physics` model defined by the user.
- `config` Configuration structure defined by the user with solvers, schemes, runtime and hardware structures configuration details.
- `limit_gradient` flag use to activate gradient limiters in the solver (default = `false`)
- `pref` Reference pressure value for cases that do not have a pressure defining BC. Incompressible solvers only (default = `nothing`)
- `ncorrectors` number of non-orthogonality correction loops (default = `0`)
- `inner_loops` number to inner loops used in transient solver based on PISO algorithm (default = `0`)

# Output

This function returns a `NamedTuple` for accessing the residuals (e.g. `residuals.Ux`) with the following entries:

- `Ux` Vector of x-velocity residuals for each iteration.
- `Uy` Vector of y-velocity residuals for each iteration.
- `Uz` Vector of y-velocity residuals for each iteration.
- `p` Vector of pressure residuals for each iteration.

"""
function simple!(
    model, config; 
    limit_gradient=false, pref=nothing, ncorrectors=0, inner_loops=0
    )

    residuals = setup_incompressible_solvers(
        SIMPLE, model, config; 
        limit_gradient=limit_gradient, 
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops
        )

    return residuals
end

# Setup for all incompressible algorithms
function setup_incompressible_solvers(
    solver_variant, model, config; 
    limit_gradient=false, pref=nothing, ncorrectors=0, inner_loops=0
    ) 

    (; solvers, schemes, runtime, hardware) = config

    @info "Extracting configuration and input fields..."

    (; U, p) = model.momentum
    mesh = model.domain

    @info "Pre-allocating fields..."
    
    ∇p = Grad{schemes.p.gradient}(p)
    mdotf = FaceScalarField(mesh)
    rDf = FaceScalarField(mesh)
    initialise!(rDf, 1.0)
    nueff = FaceScalarField(mesh)
    divHv = ScalarField(mesh)

    @info "Defining models..."

    U_eqn = (
        Time{schemes.U.time}(U)
        + Divergence{schemes.U.divergence}(mdotf, U) 
        - Laplacian{schemes.U.laplacian}(nueff, U) 
        == 
        - Source(∇p.result)
    ) → VectorEquation(mesh)

    p_eqn = (
        - Laplacian{schemes.p.laplacian}(rDf, p) == - Source(divHv)
    ) → ScalarEquation(mesh)

    @info "Initialising preconditioners..."

    @reset U_eqn.preconditioner = set_preconditioner(
                    solvers.U.preconditioner, U_eqn, U.BCs, config)
    @reset p_eqn.preconditioner = set_preconditioner(
                    solvers.p.preconditioner, p_eqn, p.BCs, config)


    @info "Pre-allocating solvers..."
     
    @reset U_eqn.solver = solvers.U.solver(_A(U_eqn), _b(U_eqn, XDir()))
    @reset p_eqn.solver = solvers.p.solver(_A(p_eqn), _b(p_eqn))

    @info "Initialising turbulence model..."
    turbulenceModel = initialise(model.turbulence, model, mdotf, p_eqn, config)

    residuals  = solver_variant(
        model, turbulenceModel, ∇p, U_eqn, p_eqn, config; 
        limit_gradient=limit_gradient, 
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops)

    return residuals
end # end function

function SIMPLE(
    model, turbulenceModel, ∇p, U_eqn, p_eqn, config; 
    limit_gradient=false, pref=nothing, ncorrectors=0, inner_loops=0
    )
    
    # Extract model variables and configuration
    (; U, p) = model.momentum
    (; nu) = model.fluid
    mesh = model.domain
    # p_model = p_eqn.model
    (; solvers, schemes, runtime, hardware) = config
    (; iterations, write_interval) = runtime
    (; backend) = hardware
    
    mdotf = get_flux(U_eqn, 2)
    nueff = get_flux(U_eqn, 3)
    rDf = get_flux(p_eqn, 1)
    divHv = get_source(p_eqn, 1)

    @info "Initialise VTKWriter (Store mesh in host memory)"

    VTKMeshData = initialise_writer(model.domain)
    
    @info "Allocating working memory..."

    # Define aux fields 
    gradU = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    Uf = FaceVectorField(mesh)
    S = StrainRate(gradU, gradUT, U, Uf)

    n_cells = length(mesh.cells)
    pf = FaceScalarField(mesh)
    # gradpf = FaceVectorField(mesh)
    Hv = VectorField(mesh)
    rD = ScalarField(mesh)

    # Pre-allocate auxiliary variables
    TF = _get_float(mesh)
    prev = zeros(TF, n_cells)
    prev = _convert_array!(prev, backend) 

    # Pre-allocate vectors to hold residuals 
    R_ux = ones(TF, iterations)
    R_uy = ones(TF, iterations)
    R_uz = ones(TF, iterations)
    R_p = ones(TF, iterations)
    
    # Initial calculations
    time = zero(TF) # assuming time=0
    interpolate!(Uf, U, config)   
    correct_boundaries!(Uf, U, U.BCs, time, config)
    flux!(mdotf, Uf, config)
    grad!(∇p, pf, p, p.BCs, time, config)
    limit_gradient && limit_gradient!(∇p, p, config)

    update_nueff!(nueff, nu, model.turbulence, config)

    @info "Starting SIMPLE loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    for iteration ∈ 1:iterations
        time = iteration

        solve_equation!(U_eqn, U, solvers.U, xdir, ydir, zdir, config)
        
        # Pressure correction
        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rDf, rD, config)
        remove_pressure_source!(U_eqn, ∇p, config)
        H!(Hv, U, U_eqn, config)
        
        # Interpolate faces
        interpolate!(Uf, Hv, config) # Careful: reusing Uf for interpolation
        correct_boundaries!(Uf, Hv, U.BCs, time, config)

        # old approach
        # div!(divHv, Uf, config) 

        # new approach
        flux!(mdotf, Uf, config)
        div!(divHv, mdotf, config)
        
        # Pressure calculations
        @. prev = p.values
        solve_equation!(p_eqn, p, solvers.p, config; ref=pref)
        explicit_relaxation!(p, prev, solvers.p.relax, config)
        
        grad!(∇p, pf, p, p.BCs, time, config) 
        limit_gradient && limit_gradient!(∇p, p, config)

        # non-orthogonal correction
        for i ∈ 1:ncorrectors
            discretise!(p_eqn, p, config)       
            apply_boundary_conditions!(p_eqn, p.BCs, nothing, time, config)
            setReference!(p_eqn, pref, 1, config)
            nonorthogonal_face_correction(p_eqn, ∇p, rDf, config)
            update_preconditioner!(p_eqn.preconditioner, p.mesh, config)
            solve_system!(p_eqn, solvers.p, p, nothing, config)
            explicit_relaxation!(p, prev, solvers.p.relax, config)
            grad!(∇p, pf, p, p.BCs, time, config) 
            limit_gradient && limit_gradient!(∇p, p, config)
        end

        # Velocity and boundaries correction

        # old approach
        # correct_velocity!(U, Hv, ∇p, rD, config)
        # interpolate!(Uf, U, config)
        # correct_boundaries!(Uf, U, U.BCs, time, config)
        # flux!(mdotf, Uf, config) 

        # new approach

        # 1. using velocity from momentum equation
        interpolate!(Uf, U, config)
        correct_boundaries!(Uf, U, U.BCs, time, config)
        flux!(mdotf, Uf, config)
        correct_mass_flux(mdotf, p, rDf, config)
        # correct_mass_flux2(mdotf, p_eqn, p, config)
        correct_velocity!(U, Hv, ∇p, rD, config)

        turbulence!(turbulenceModel, model, S, prev, time, limit_gradient, config) 
        update_nueff!(nueff, nu, model.turbulence, config)
        
        convergence = 1e-7

        residual!(R_ux, U_eqn, U.x, iteration, xdir, config)
        residual!(R_uy, U_eqn, U.y, iteration, ydir, config)
        typeof(mesh) <: Mesh3 && residual!(R_uz, U_eqn, U.z, iteration, zdir, config)
        residual!(R_p, p_eqn, p, iteration, nothing, config)

        if (R_ux[iteration] <= convergence && 
            R_uy[iteration] <= convergence && 
            R_uz[iteration] <= convergence &&
            R_p[iteration] <= convergence)

            @info "Simulation converged in $iteration iterations!"
                if !signbit(write_interval)
                    model2vtk(model, @sprintf "iteration_%.6d" iteration)
                end
            break
        end

        ProgressMeter.next!(
            progress, showvalues = [
                (:iter,iteration),
                (:Ux, R_ux[iteration]),
                (:Uy, R_uy[iteration]),
                (:Uz, R_uz[iteration]),
                (:p, R_p[iteration]),
                ]
            )

        if iteration%write_interval + signbit(write_interval) == 0      
            model2vtk(model, VTKMeshData, @sprintf "iteration_%.6d" iteration)
        end

    end # end for loop
    
    return (Ux=R_ux, Uy=R_uy, Uz=R_uz, p=R_p)
end

### TEMP LOCATION FOR PROTOTYPING - NONORTHOGONAL CORRECTION 

function nonorthogonal_face_correction(eqn, grad, flux, config)
    mesh = grad.mesh
    (; faces, cells, boundary_cellsID) = mesh

    (; hardware) = config
    (; backend, workgroup) = hardware

    (; b) = eqn.equation
    
    n_faces = length(faces)
    n_bfaces = length(boundary_cellsID)
    n_ifaces = n_faces - n_bfaces

    kernel! = _nonorthogonal_face_correction(backend, workgroup)
    kernel!(b, grad, flux, faces, cells, n_bfaces, ndrange=n_ifaces)
    KernelAbstractions.synchronize(backend)
end

@kernel function _nonorthogonal_face_correction(b, grad, flux, faces, cells, n_bfaces)
    i = @index(Global)
    fID = i + n_bfaces
    face = faces[fID]
    (; ownerCells, area, normal, e, delta) = face
    cID1 = ownerCells[1]
    cID2 = ownerCells[2]
    # cell1 = cells[cID1]
    # cell2 = cells[cID2]

    (; values) = grad.field
    w, df = weight(cells, faces, fID)
    gradi = w*grad[cID1] + (1 - w)*grad[cID2]
    gradf = gradi + ((values[cID2] - values[cID1])/delta - (gradi⋅e))*e


    Sf = area*normal
    Ef = ((Sf⋅Sf)/(Sf⋅e))*e
    T_hat = Sf - Ef
    faceCorrection = flux[fID]*gradf⋅T_hat

    Atomix.@atomic b[cID1] -= faceCorrection#*cell1.volume
    Atomix.@atomic b[cID2] += faceCorrection#*cell2.volume # should this be -ve?
        
end

function weight(cells, faces, fi)
    (; ownerCells, centre) = faces[fi]
    cID1 = ownerCells[1]
    cID2 = ownerCells[2]
    c1 = cells[cID1].centre
    c2 = cells[cID2].centre
    c1_f = centre - c1
    c1_c2 = c2 - c1
    q = (c1_f⋅c1_c2)/(c1_c2⋅c1_c2)
    f_prime = c1 - q*(c1 - c2)
    w = norm(c2 - f_prime)/norm(c2 - c1)
    df = centre - f_prime
    return w, df
end

### TEMP LOCATION FOR PROTOTYPING

function correct_mass_flux(mdotf, p, rDf, config)
    # sngrad = FaceScalarField(mesh)
    (; faces, cells, boundary_cellsID) = mdotf.mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    n_faces = length(faces)
    n_bfaces = length(boundary_cellsID)
    n_ifaces = n_faces - n_bfaces #+ 1

    kernel! = _correct_mass_flux(backend, workgroup)
    kernel!(mdotf, p, rDf, faces, cells, n_bfaces, ndrange=length(n_ifaces))
    KernelAbstractions.synchronize(backend)
end

@kernel function _correct_mass_flux(mdotf, p, rDf, faces, cells, n_bfaces)
    i = @index(Global)
    fID = i + n_bfaces

    @inbounds begin 
        face = faces[fID]
        (; area, normal, ownerCells, delta) = face 
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        p1 = p[cID1]
        p2 = p[cID2]
        face_grad = area*(p2 - p1)/delta # best option so far!
        mdotf[fID] -= face_grad*rDf[fID]
    end
end