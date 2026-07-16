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


function loss_alg2(
    x::Vector{<:CuArray{Float32, 4}}, #length M vector of 4d target tensors (i, j, channel vector, R = number of samples), here R=1
    y::Vector{<:CuArray{Float32, 3}} #length M of 3d target tensors (i, j, channel vector)
    )
    @assert length(x) == length(y)
    total_loss = 0
    M = size(y, 1)
    #R = size(x[1], 4)
    
    for m in 1:M
        #x[m-1] is 4d tensor of batch of R samples at checkpoint m-1, expected for loss_fn
        #pred_x = phi_n(phi, x[m], theta, t)
        total_loss += loss_fn(x[m], y[m])
    end
    
    total_loss = total_loss/M
    return total_loss
end
