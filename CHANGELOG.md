# Release notes

The format used for this `changelog` is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Notice that until the package reaches version `v1.0.0` minor releases are likely to be `breaking`. Starting from version `v3.0.1` breaking changes will be recorded here. 

## Version [v0.3.1] - 2024-10-18

### Added
* Vastly improved documentation with new examples provided
* Changelog added to record changes more clearly. Record kept in [Release notes](@ref)

### Fixed
* The calculation of gradients can be limited for stability. This functionality can be activated by passing the key work argument `limit_gradient` to the `run!` function. The implementation has been improved for robustness.

### Changed
* Master branch protected and requires PRs to push changes

### Breaking
* No breaking changes

### Deprecated
* No functions deprecated

### Removed
* No functionality has been removed

## Version [v0.3.0] - 2024-09-21

* New name - XCALibre.jl - which is now registered in the General Julia registry
* Can do 3D and GPU accelerated simulations
* Can read .unv and OpenFOAM mesh files (3D)
* Can do incompressible and compressible simulations
* RANS and LES models available
* User-provided functions or neural networks for boundary conditions
* Reasonably complete "user" documentation now provided
* Made repository public (in v0.2 the work was kept in a private repository and could only do 2D simulations)
* Tidy up mesh type definitions by @mberto79 in #5
* Adapt code base to work with new mesh format by @mberto79 in [#6]
* Mesh boundary struct changes PR by @TomMazin in [#7]
* Mesh boundary struct changes PR fix by @TomMazin in [#8]


## Version [v0.2.0] - 2023-01-23

* New mesh format and type implemented that are GPU friendly.
* No functionality changes

## Version [v0.1.0] - 2023-01-23

### Initial release

2D implementation of classic incompressible solvers for laminar and turbulent flows:

* Framework for equation definition
* SIMPLE nad PISO algorithms
* Read UNV meshes in 2D
* Capability for RANS models
* Various discretisation schemes available
* Planned extension to 3D and GPU acceleration!