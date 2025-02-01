export JacobiSmoother

abstract type AbstractSmoother end

"""
    struct JacobiSmoother{L,F,V} <: AbstractSmoother
        loops::L
        omega::F
        x_temp::V
    end

Structure to hold information for using the weighted Jacobi smoother. 

# Fields
- `loops` represents the number of smoothing iterations.
- `omega` represents the relaxation weight, 1 corresponds to no weighting. Typically a weight of 2/3 is used to filter high frequencies in the residual field.
- `x_temp` is a vector used internally to store intermediate solutions.

# Constructors

    JacobiSmoother(mesh::AbstractMesh)

This constructor takes a mesh object as input to create the smoother. The number of cells in the mesh is used to allocate `x_temp`. The number of loops and smoother factor are set to 5 and 1, respectively.

    JacobiSmoother(; domain, loops, omega=2/3)

Alternative constructor for `JacobiSmoother`.

# keyword arguments
- `domain` represents a mesh object of type `AbstractMesh`.
- `loops` is the number of iterations to be used
- `omega` represents the weighting factor, 1 does not relax the system, 2/3 is found to work well for smoothing high frequencies in the residual field
"""
struct JacobiSmoother{L<:Integer,F<:Number,V<:AbstractArray} <: AbstractSmoother
    loops::L
    omega::F
    x_temp::V
end
Adapt.@adapt_structure JacobiSmoother

JacobiSmoother(mesh::AbstractMesh) = begin
    x = zeros(_get_float(mesh), length(mesh.cells))
    backend = _get_backend(mesh)
    JacobiSmoother(5, one(_get_int(mesh)), adapt(backend, x))
end

JacobiSmoother(; domain, loops, omega=2/3) = begin
    x = zeros(_get_float(domain), length(domain.cells))
    backend = _get_backend(domain)
    F = _get_float(domain)
    JacobiSmoother(loops, F(omega), adapt(backend, x))
end

apply_smoother!(smoother::Nothing, x, A, b, hardware) = nothing

function apply_smoother!(smoother::AbstractSmoother, x, A, b, hardware)
    (; backend, workgroup) = hardware

    nzval = _nzval(A)
    colval = _colval(A)
    rowptr = _rowptr(A)

    _smoother_launch(backend, smoother, x, b, nzval, colval, rowptr, workgroup)
end

# CPU implementation
function _smoother_launch(
    backend::CPU, smoother, x, b, nzval, colval, rowptr, workgroup
    )
    _apply_smoother!(smoother, x, b, nzval, colval, rowptr)
end

function _apply_smoother!(
    smoother::JacobiSmoother, x, b, nzval, colval, rowptr
    )
    ω = smoother.omega
    for _ ∈ 1:smoother.loops
        Threads.@threads for cID ∈ eachindex(x)
            cIndex =  spindex(rowptr, colval, cID, cID)
            xi = multiply_row(x, b, ω, nzval, rowptr, colval,  cID, cIndex)
            smoother.x_temp[cID] = xi
        end
        Threads.@threads for cID ∈ eachindex(smoother.x_temp)
            @inbounds x[cID] = smoother.x_temp[cID]
        end
    end
end

# GPU Kernel
function _smoother_launch(
    backend::GPU, smoother, x, b, nzval, colval, rowptr, workgroup
    )
    krange = length(x)
    kernel! = _apply_smoother_gpu!(backend, workgroup)
    for _ ∈ 1:smoother.loops
        kernel!(smoother, x, b, nzval, colval, rowptr, ndrange = krange)
        # KernelAbstractions.synchronize(backend)
        x .= smoother.x_temp
    end
end

@kernel function _apply_smoother_gpu!(
    smoother::JacobiSmoother, x, b, nzval, colval, rowptr
    )
    cID = @index(Global)
    ω = smoother.omega
    cIndex =  spindex(rowptr, colval, cID, cID)
    xi = multiply_row(x, b, ω, nzval, rowptr, colval,  cID, cIndex)
    smoother.x_temp[cID] = xi
end

# Weighted Jacobi implementation
@inline function multiply_row(x, b, ω, nzval, rowptr, colval,  cID, cIndex)
    sum = zero(eltype(nzval))
    uno = one(eltype(colval))
    @inbounds begin
        for nzvali ∈ rowptr[cID]:(rowptr[cID+1] - 1)
            j = colval[nzvali]
            xj = x[j] 
            sum += nzval[nzvali]*xj
        end
        a_ii = nzval[cIndex]
        xi = x[cID]
        sum -= a_ii*xi # remove multiplication with diagonal (faster than "if")
        rD = one(eltype(colval))/a_ii
        return ω*rD*(b[cID] - sum) + (uno - ω)*xi
    end
end