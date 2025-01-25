using XCALibre
using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using Test

TEST_CASES_DIR = pkgdir(XCALibre, "test/0_TEST_CASES")

@testset verbose = true "Functionality tests" begin

    @testset "Mesh conversion" begin
        include("test_mesh_conversion.jl")
    end

    @testset "Smoothers" begin
        include("test_smoothers.jl")
    end

    @testset "DILU" begin
        include("test_DILU.jl")
    end

    @testset "Incompressible" begin

        test_files = [
            "2d_incompressible_flatplate_KOmega_lowRe.jl",
            "2d_incompressible_flatplate_KOmega_HighRe.jl",
            "2d_incompressible_laminar_BFS.jl",
            "2d_incompressible_transient_KOmega_BFS_lowRe.jl",
            "2d_incompressible_transient_laminar_BFS.jl",
            "3d_incompressible_laminar_BFS.jl",
            "3d_incompressible_laminar_cascade_periodic.jl"
        ]

        for test ∈ test_files
            test_path = joinpath(TEST_CASES_DIR, test)
            include(test_path)
        end
    end

    @testset "Compressible" begin
        test_files = [
            "2d_compressible_KOmega_flatplate_fixedT.jl",
            "2d_compressible_laminar_flatplate_fixedT.jl",
            "2d_compressible_transient_laminar_heated_cylinder.jl"
        ]

        for test ∈ test_files
            test_path = joinpath(TEST_CASES_DIR, test)
            include(test_path)
        end
    end

    foreach(rm, filter(endswith(".vtk"), readdir(pwd(), join=true)))
    foreach(rm, filter(endswith(".vtu"), readdir(pwd(), join=true)))

end # end "functionality test"