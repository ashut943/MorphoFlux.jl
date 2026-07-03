function loss_fn(x::CuArray{Float32,4}, target::CuArray{Float32,3}; vc::Int = 4)
    vis  = x[:, :, 1:vc, :]
    tgt  = reshape(target[:, :, 1:vc], size(target, 1), size(target, 2), vc, 1)
    return mean((vis .- tgt) .^ 2)
end

function trajectory_loss(
    xs::Vector{<:CuArray{Float32,4}},
    targets::Vector{<:CuArray{Float32,3}},
    weights::Vector{Float32} = ones(Float32, length(xs));
    vc::Int = 4,
)
    @assert length(xs) == length(targets) == length(weights)
    total = 0f0
    for (x, tgt, w) in zip(xs, targets, weights)
        vis = x[:, :, 1:vc, :]
        diff = vis .- reshape(tgt, size(tgt)..., 1)
        total += w * mean(diff .^ 2)
    end
    return total
end

function per_sample_loss(x::CuArray{Float32,4}, target::CuArray{Float32,3}; vc::Int = 4)
    vis  = x[:, :, 1:vc, :]
    tgt  = reshape(target[:, :, 1:vc], size(target, 1), size(target, 2), vc, 1)
    return Array(vec(mean((vis .- tgt) .^ 2; dims=(1, 2, 3))))
end
