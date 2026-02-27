export REF_FRAME

struct REF_FRAME_values
    type::Int64
    omega::Float64
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
