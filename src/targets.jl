function image_to_nca_target(img, W::Int, H::Int)
    src_h, src_w = size(img)
    target = zeros(Float32, W, H, 4)

    @inbounds for y in 1:H, x in 1:W
        y0 = floor(Int, (y - 1) * src_h / H) + 1
        y1 = max(y0, floor(Int, y * src_h / H))
        x0 = floor(Int, (x - 1) * src_w / W) + 1
        x1 = max(x0, floor(Int, x * src_w / W))

        rsum = gsum = bsum = asum = 0f0
        count = 0
        for sy in y0:y1, sx in x0:x1
            px = img[sy, sx]
            a = Float32(alpha(px))
            rsum += Float32(red(px)) * a
            gsum += Float32(green(px)) * a
            bsum += Float32(blue(px)) * a
            asum += a
            count += 1
        end
        invc = 1f0 / Float32(count)
        target[x, y, 1] = rsum * invc
        target[x, y, 2] = gsum * invc
        target[x, y, 3] = bsum * invc
        target[x, y, 4] = asum * invc
    end
    target
end

function load_target_rgba(path::String; target_size::Int = 40)
    isfile(path) || error("no such file: $path")
    if endswith(path, ".jld2")
        target = JLD2.load(path, "target")
    elseif endswith(lowercase(path), ".png")
        target = image_to_nca_target(load(path), target_size, target_size)
    else
        error("want .jld2 or .png, got $path")
    end
    cu(Float32.(target))
end

function make_seed(W::Int, H::Int, channel_n::Int; cx::Int = W ÷ 2 + 1, cy::Int = H ÷ 2 + 1)
    (1 <= cx <= W && 1 <= cy <= H) || error("seed ($cx,$cy) outside grid $(W)x$(H)")
    seed = CUDA.zeros(Float32, W, H, channel_n)
    cpu = zeros(Float32, W, H, channel_n)
    cpu[cx, cy, 4:end] .= 1f0
    copyto!(seed, cu(cpu))
    seed
end
