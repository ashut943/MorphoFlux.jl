struct GenomeCAModel
    encoder_w1::CuArray{Float32,2}
    encoder_b1::CuArray{Float32,1}
    encoder_mu_w::CuArray{Float32,2}
    encoder_mu_b::CuArray{Float32,1}
    encoder_logvar_w::CuArray{Float32,2}
    encoder_logvar_b::CuArray{Float32,1}
    hyper_w1::CuArray{Float32,2}
    hyper_b1::CuArray{Float32,1}
    hyper_w2::CuArray{Float32,2}
    hyper_b2::CuArray{Float32,1}
    base_theta::CuArray{Float32,1}
    channel_n::Int
    hidden_n::Int
    target_w::Int
    target_h::Int
    latent_k::Int
    hyper_output_scale::Float32
    fire_rate::Float32
end

Flux.@layer GenomeCAModel

Flux.trainable(m::GenomeCAModel) = (;
    encoder_w1       = m.encoder_w1,
    encoder_b1       = m.encoder_b1,
    encoder_mu_w     = m.encoder_mu_w,
    encoder_mu_b     = m.encoder_mu_b,
    encoder_logvar_w = m.encoder_logvar_w,
    encoder_logvar_b = m.encoder_logvar_b,
    hyper_w1         = m.hyper_w1,
    hyper_b1         = m.hyper_b1,
    hyper_w2         = m.hyper_w2,
    hyper_b2         = m.hyper_b2,
)

function GenomeCAModel(hp::GenomeHParams)
    C = hp.channel_n
    H = hp.hidden_n
    K = hp.genome_k
    W = hp.target_size + 2hp.target_padding
    target_input_n = W * W * 4
    theta_n = generated_theta_size(C, H)

    encoder_w1 = cu(Float32.(randn(hp.encoder_hidden_n, target_input_n) .* sqrt(2f0 / target_input_n)))
    encoder_b1 = CUDA.zeros(Float32, hp.encoder_hidden_n)
    encoder_mu_w = cu(Float32.(randn(K, hp.encoder_hidden_n) .* 0.01f0))
    encoder_mu_b = CUDA.zeros(Float32, K)
    encoder_logvar_w = cu(Float32.(randn(K, hp.encoder_hidden_n) .* 0.01f0))
    encoder_logvar_b = CUDA.zeros(Float32, K)

    hyper_w1 = cu(Float32.(randn(hp.hyper_hidden_n, K) .* sqrt(2f0 / K)))
    hyper_b1 = CUDA.zeros(Float32, hp.hyper_hidden_n)
    # small residuals on top of a fixed base theta — full hyper bias alone tends to blow up
    hyper_w2 = CUDA.zeros(Float32, theta_n, hp.hyper_hidden_n)
    hyper_b2 = CUDA.zeros(Float32, theta_n)
    base_theta = initial_theta_vector(C, H)

    return GenomeCAModel(
        encoder_w1, encoder_b1,
        encoder_mu_w, encoder_mu_b,
        encoder_logvar_w, encoder_logvar_b,
        hyper_w1, hyper_b1, hyper_w2, hyper_b2, base_theta,
        C, H, W, W, K, hp.hyper_output_scale, hp.fire_rate,
    )
end

function generated_theta_size(channel_n::Int, hidden_n::Int)
    return 3channel_n * hidden_n + hidden_n + hidden_n * channel_n + channel_n
end

function initial_theta_vector(channel_n::Int, hidden_n::Int)
    dense1_w = Float32.(randn(1, 1, 3channel_n, hidden_n) .* sqrt(2f0 / (3channel_n)))
    dense1_b = zeros(Float32, hidden_n)
    dense2_w = zeros(Float32, 1, 1, hidden_n, channel_n)
    dense2_b = zeros(Float32, channel_n)
    return cu(vcat(vec(dense1_w), dense1_b, vec(dense2_w), dense2_b))
end

function encode_targets(model::GenomeCAModel, targets::CuArray{Float32,4})
    _, _, _, B = size(targets)
    flat = reshape(targets, :, B)
    h = relu.(model.encoder_w1 * flat .+ reshape(model.encoder_b1, :, 1))
    mu = model.encoder_mu_w * h .+ reshape(model.encoder_mu_b, :, 1)
    logvar = model.encoder_logvar_w * h .+ reshape(model.encoder_logvar_b, :, 1)
    return mu, logvar
end

function sample_latents(mu::CuArray{Float32,2}, logvar::CuArray{Float32,2}, eps::CuArray{Float32,2})
    return mu .+ exp.(0.5f0 .* logvar) .* eps
end

function latent_kl_loss(mu::CuArray{Float32,2}, logvar::CuArray{Float32,2})
    return -0.5f0 * mean(1f0 .+ logvar .- mu .^ 2 .- exp.(logvar))
end

function hypernetwork_theta(model::GenomeCAModel, g::CuArray{Float32,2})
    h = relu.(model.hyper_w1 * g .+ reshape(model.hyper_b1, :, 1))
    residual = model.hyper_w2 * h .+ reshape(model.hyper_b2, :, 1)
    return reshape(model.base_theta, :, 1) .+ model.hyper_output_scale .* residual
end

struct GeneratedNCAWeights
    dense1_w::CuArray{Float32,4}
    dense1_b::CuArray{Float32,1}
    dense2_w::CuArray{Float32,4}
    dense2_b::CuArray{Float32,1}
end

function unpack_generated_theta(model::GenomeCAModel, theta_col::CuArray{Float32,2})
    C = model.channel_n
    H = model.hidden_n
    n_dense1_w = 3C * H
    n_dense1_b = H
    n_dense2_w = H * C

    p1 = n_dense1_w
    p2 = p1 + n_dense1_b
    p3 = p2 + n_dense2_w

    dense1_w = reshape(theta_col[1:p1, :], 1, 1, 3C, H)
    dense1_b = vec(theta_col[p1+1:p2, :])
    dense2_w = reshape(theta_col[p2+1:p3, :], 1, 1, H, C)
    dense2_b = vec(theta_col[p3+1:p3+C, :])
    return GeneratedNCAWeights(dense1_w, dense1_b, dense2_w, dense2_b)
end

function step_generated_nca(
    model::GenomeCAModel,
    x::CuArray{Float32,4},
    kernel::CuArray{Float32,4},
    theta::GeneratedNCAWeights;
    fire_rate::Float32 = model.fire_rate,
    step_size::Float32 = 1f0,
    update_mask = nothing,
)
    pre_life = NCA.get_living_mask(x)
    y = NCA.perceive(x, kernel)

    h = NNlib.conv(y, theta.dense1_w) .+ reshape(theta.dense1_b, 1, 1, :, 1)
    h = relu.(h)
    dx = NNlib.conv(h, theta.dense2_w) .+ reshape(theta.dense2_b, 1, 1, :, 1)
    dx = dx .* step_size

    mask = update_mask === nothing ?
        Float32.(CUDA.rand(Float32, size(x, 1), size(x, 2), 1, size(x, 4)) .<= fire_rate) :
        Zygote.dropgrad(update_mask)
    x_new = x .+ dx .* mask

    post_life = NCA.get_living_mask(x_new)
    life = pre_life .* post_life
    return x_new .* life
end

function genome_loss_fn(x::CuArray{Float32,4}, target_batch::CuArray{Float32,4})
    rgba = x[:, :, 1:4, :]
    diff = rgba .- target_batch
    return mean(diff .^ 2)
end

function per_sample_loss_single_target(x::CuArray{Float32,4}, target::CuArray{Float32,3})
    W, H = size(target, 1), size(target, 2)
    rgba = x[:, :, 1:4, :]
    diff = rgba .- reshape(target[:, :, 1:4], W, H, 4, 1)
    per_sample = mean(diff .^ 2; dims=(1, 2, 3))
    return Array(vec(per_sample))
end
