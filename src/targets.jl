_to_rgba(px::ColorTypes.AbstractRGBA) = px
_to_rgba(px::ColorTypes.AbstractRGB)  = ColorTypes.RGBA(red(px), green(px), blue(px), 1f0)
_to_rgba(px::ColorTypes.AGray)        = (g = Float32(gray(px)); ColorTypes.RGBA(g, g, g, Float32(alpha(px))))
_to_rgba(px::ColorTypes.GrayA)        = (g = Float32(gray(px)); ColorTypes.RGBA(g, g, g, Float32(alpha(px))))
_to_rgba(px::ColorTypes.AbstractGray) = (g = Float32(gray(px)); ColorTypes.RGBA(g, g, g, 1f0))

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
            px = _to_rgba(img[sy, sx])
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

"""
Two-channel Grow-style target in a `(W,H,4)` array (channels 3–4 unused):

- channel 1: premultiplied gray `lum * alpha`
- channel 2: alpha (`1 - lum` from linear luminance), so bright background → transparent

Derived from the same premultiplied RGBA grid as `image_to_nca_target`. Best for **grayscale**
art (e.g. black ink on white); colored PNGs should use full `image_to_nca_target` + `vc=4`.
"""
function image_to_nca_target_mono2(img, W::Int, H::Int)
    t = image_to_nca_target(img, W, H)
    r = @view t[:, :, 1]
    g = @view t[:, :, 2]
    b = @view t[:, :, 3]
    a = @view t[:, :, 4]
    inva = @. ifelse(a > 1f-6, 1f0 / a, 0f0)
    lin_r = @. clamp(r * inva, 0f0, 1f0)
    lin_g = @. clamp(g * inva, 0f0, 1f0)
    lin_b = @. clamp(b * inva, 0f0, 1f0)
    lum = @. (lin_r + lin_g + lin_b) / 3f0
    life_a = @. clamp(1f0 - lum, 0f0, 1f0)
    premul_gray = @. clamp(lum * life_a, 0f0, 1f0)
    out = zeros(Float32, W, H, 4)
    out[:, :, 1] .= premul_gray
    out[:, :, 2] .= life_a
    return out
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

"""Which visible channel carries alpha for seeding (matches `step` life_ch when life mask on)."""
function seed_life_channel(visible_channels::Int, channel_n::Int)
    visible_channels == 2 && return min(2, channel_n)
    visible_channels == 4 && return min(4, channel_n)
    return min(4, channel_n) # vc=1, 3, or default: legacy channel-4 alpha seed
end

function make_seed(
    W::Int,
    H::Int,
    channel_n::Int;
    cx::Int = W ÷ 2 + 1,
    cy::Int = H ÷ 2 + 1,
    visible_channels::Int = min(4, channel_n),
)
    (1 <= cx <= W && 1 <= cy <= H) || error("seed ($cx,$cy) outside grid $(W)x$(H)")
    seed = CUDA.zeros(Float32, W, H, channel_n)
    cpu = zeros(Float32, W, H, channel_n)
    life_ch = seed_life_channel(visible_channels, channel_n)
    cpu[cx, cy, life_ch:end] .= 1f0
    copyto!(seed, cu(cpu))
    seed
end
