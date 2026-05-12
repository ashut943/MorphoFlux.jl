function loss_fn(x::CuArray{Float32,4}, target::CuArray{Float32,3})
    rgba = x[:, :, 1:4, :]
    diff = rgba .- reshape(target, size(target)..., 1)
    per_sample = mean(diff .^ 2; dims=(1, 2, 3))
    return mean(per_sample)
end

function per_sample_loss(x::CuArray{Float32,4}, target::CuArray{Float32,3})
    rgba = x[:, :, 1:4, :]
    diff = rgba .- reshape(target, size(target)..., 1)
    per_sample = mean(diff .^ 2; dims=(1, 2, 3))
    return Array(vec(per_sample))
end
