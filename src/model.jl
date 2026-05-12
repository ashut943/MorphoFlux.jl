"""
Build the fixed 3x3 perception kernel.

Returns a CuArray of shape (3, 3, channel_n, 3*channel_n) suitable for
`NNlib.conv`. Each input channel is convolved with 3 filters:
identity, dx (Sobel-x), dy (Sobel-y), optionally rotated by `angle`.
"""
function build_perception_kernel(channel_n::Int; angle::Float32 = 0f0)
    ident = Float32[0 0 0; 0 1 0; 0 0 0]
    sx = Float32[-1 0 1; -2 0 2; -1 0 1] ./ 8f0
    sy = permutedims(sx)

    c, s = cos(angle), sin(angle)
    dx = c .* sx .- s .* sy
    dy = s .* sx .+ c .* sy

    filters = cat(ident, dx, dy; dims=3)
    kernel = zeros(Float32, 3, 3, channel_n, 3channel_n)
    for c in 1:channel_n
        out_base = 3 * (c - 1)
        kernel[:, :, c, out_base+1:out_base+3] .= filters
    end

    return kernel |> cu
end

function perceive(x::CuArray{Float32,4}, kernel::CuArray{Float32,4})
    return NNlib.conv(x, kernel; pad=1)
end

struct CAModel
    dense1_w::CuArray{Float32,4}
    dense1_b::CuArray{Float32,1}
    dense2_w::CuArray{Float32,4}
    dense2_b::CuArray{Float32,1}
    channel_n::Int
    fire_rate::Float32
end

Flux.@layer CAModel

Flux.trainable(m::CAModel) = (;
    dense1_w = m.dense1_w,
    dense1_b = m.dense1_b,
    dense2_w = m.dense2_w,
    dense2_b = m.dense2_b,
)

function CAModel(hp::HParams)
    C = hp.channel_n
    H = hp.hidden_n
    dense1_w = cu(Float32.(randn(1, 1, 3C, H) .* sqrt(2f0 / (3C))))
    dense1_b = CUDA.zeros(Float32, H)
    dense2_w = CUDA.zeros(Float32, 1, 1, H, C)
    dense2_b = CUDA.zeros(Float32, C)
    return CAModel(dense1_w, dense1_b, dense2_w, dense2_b, C, hp.fire_rate)
end

function living_mask_kernel!(out, x, W::Int, H::Int, N::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    total = W * H * N

    while idx <= total
        z = idx - 1
        ix = z % W + 1
        iy = (z ÷ W) % H + 1
        ib = z ÷ (W * H) + 1

        alive = false
        for dy in -1:1
            yy = iy + dy
            if 1 <= yy <= H
                for dx in -1:1
                    xx = ix + dx
                    if 1 <= xx <= W && x[xx, yy, 4, ib] > 0.1f0
                        alive = true
                    end
                end
            end
        end

        out[ix, iy, 1, ib] = alive ? 1f0 : 0f0
        idx += stride
    end

    return nothing
end

function get_living_mask(x::CuArray{Float32,4})
    W, H, _, N = size(x)
    out = similar(x, Float32, W, H, 1, N)
    threads = 256
    blocks = cld(W * H * N, threads)
    CUDA.@cuda threads=threads blocks=blocks living_mask_kernel!(out, x, W, H, N)
    return out
end

Zygote.@non_differentiable get_living_mask(x)

function step(
    model::CAModel,
    x::CuArray{Float32,4},
    kernel::CuArray{Float32,4};
    fire_rate::Float32 = model.fire_rate,
    step_size::Float32 = 1f0,
    update_mask = nothing,
)
    pre_life = get_living_mask(x)
    y = perceive(x, kernel)

    dx = NNlib.conv(y, model.dense1_w) .+ reshape(model.dense1_b, 1, 1, :, 1)
    dx = relu.(dx)
    dx = NNlib.conv(dx, model.dense2_w) .+ reshape(model.dense2_b, 1, 1, :, 1)
    dx = dx .* step_size

    mask = update_mask === nothing ?
        Float32.(CUDA.rand(Float32, size(x, 1), size(x, 2), 1, size(x, 4)) .<= fire_rate) :
        update_mask
    x_new = x .+ dx .* mask

    post_life = get_living_mask(x_new)
    life = pre_life .* post_life
    return x_new .* life
end
