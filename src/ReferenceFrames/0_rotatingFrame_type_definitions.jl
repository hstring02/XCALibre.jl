export SRF
export MRF

struct SRF_values
    omega
    rotaxis
    x0
end

SRF(omega, rotaxis, x0) = begin
    SRF_values(
        omega,
        rotaxis,
        x0
    )
end

struct MRF_values
    omega
    rotaxis
    x0
    zones
end

MRF(omega, rotaxis, x0, zones) = begin
    MRF_values(
        omega,
        rotaxis,
        x0,
        zones
    )
end