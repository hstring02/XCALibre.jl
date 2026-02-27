export REF_FRAME

struct REF_FRAME_values
    type
    omega
    rotaxis
    x0
end

REF_FRAME(type, omega, rotaxis, x0) = begin
    REF_FRAME_values(
        type,
        omega,
        rotaxis,
        x0
    )
end
