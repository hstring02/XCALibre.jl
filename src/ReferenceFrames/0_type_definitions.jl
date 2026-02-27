export MRF
export SRF

struct MRF{omega, rotaxis, x0, zones}
    omega::omega
    rotaxis::rotaxis
    x0::x0
    zones::zones
end

struct SRF{omega, rotaxis, x0}
    omega::omega
    rotaxis::rotaxis
    x0::x0
end
