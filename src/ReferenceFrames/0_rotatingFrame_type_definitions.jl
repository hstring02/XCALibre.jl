export REF_FRAME

struct REF_FRAME
    type
    omega
    rotaxis
    x0
    zones
end

REF_FRAME(type, omega, rotaxis, x0, zones) = begin
    SRF_values(
        type,
        omega,
        rotaxis,
        x0,
        zones
    )
end
