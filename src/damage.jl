function make_circle_masks(n::Int, W::Int, H::Int)
    xs = reshape(range(-1f0, 1f0; length=W), :, 1)
    ys = reshape(range(-1f0, 1f0; length=H), 1, :)
    masks = zeros(Float32, W, H, 1, n)
    for k in 1:n
        cx = rand(Float32) - 0.5f0
        cy = rand(Float32) - 0.5f0
        r  = 0.1f0 + rand(Float32) * 0.3f0
        d2 = ((xs .- cx) ./ r) .^ 2 .+ ((ys .- cy) ./ r) .^ 2
        masks[:, :, 1, k] .= Float32.(d2 .< 1f0)
    end
    return cu(masks)
end

function make_update_masks(iter_n::Int, W::Int, H::Int, batch_size::Int, fire_rate::Float32)
    raw = rand(Float32, W, H, 1, batch_size, iter_n)
    return Float32.(raw .<= fire_rate)
end

function fill_update_masks!(buf::CuArray{Float32,5}, iter_n::Int, fire_rate::Float32)
    mask_view = @view(buf[:, :, :, :, 1:iter_n])
    rand!(mask_view)
    @. mask_view = Float32(mask_view <= fire_rate)
    return mask_view
end
