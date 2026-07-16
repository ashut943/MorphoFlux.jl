run_timestamp() = Dates.format(Dates.now(), dateformat"yyyymmdd_HHMMSS")

Base.@kwdef struct HParams
    channel_n::Int = 16
    hidden_n::Int = 128
    fire_rate::Float32 = 0.5f0
    target_size::Int = 40
    target_padding::Int = 16
    batch_size::Int = 8
    pool_size::Int = 1024
    train_steps::Int = 8000
    min_steps::Int = 64
    max_steps::Int = 96
    lr::Float32 = 2f-3
    lr_decay_step::Int = 2000
    lr_decay_factor::Float32 = 0.1f0
    damage_n::Int = 0
    use_pattern_pool::Bool = false
    # subset of: :id, :sobel_x, :sobel_y, :avg, :laplacian
    filters::Vector{Symbol} = [:id, :sobel_x, :sobel_y]
    # 1 = intensity only, 2 = gray+alpha (life on ch2), 4 = RGBA (life on ch4)
    visible_channels::Int = 4

    function HParams(channel_n, hidden_n, fire_rate, target_size, target_padding,
                     batch_size, pool_size, train_steps, min_steps, max_steps,
                     lr, lr_decay_step, lr_decay_factor, damage_n, use_pattern_pool,
                     filters, visible_channels)
        channel_n >= 2 || error("channel_n must be >= 2")
        visible_channels <= channel_n || error("visible_channels ($visible_channels) must be <= channel_n ($channel_n)")
        new(channel_n, hidden_n, fire_rate, target_size, target_padding,
            batch_size, pool_size, train_steps, min_steps, max_steps,
            lr, lr_decay_step, lr_decay_factor, damage_n, use_pattern_pool,
            filters, visible_channels)
    end
end


Base.@kwdef struct TrajectoryHParams
    channel_n::Int     = 16
    hidden_n::Int      = 128
    fire_rate::Float32 = 0.5f0
    # CA steps corresponding to each observation time
    # e.g. [0, 20, 45, 80] means you have 4 observations
    obs_steps::Vector{Int} = [0, 20, 45, 80]
    batch_size::Int    = 4
    train_steps::Int   = 4000
    lr::Float32        = 2f-3
    lr_decay_step::Int = 2000
    lr_decay_factor::Float32 = 0.1f0
    target_padding::Int = 0
    # subset of: :id, :sobel_x, :sobel_y, :avg, :laplacian
    filters::Vector{Symbol} = [:id, :sobel_x, :sobel_y]
    # 1 = intensity only, 2 = gray+alpha (life on ch2), 4 = RGBA (life on ch4)
    visible_channels::Int = 4
end
function scheduled_learning_rate(
    step_i::Int, train_steps::Int, lr::Float32, lr_decay_step::Int, lr_decay_factor::Float32,
)
    step_i < lr_decay_step && return lr
    lr_min = lr * lr_decay_factor
    decay_steps = max(1, train_steps - lr_decay_step)
    t = Float32(clamp((step_i - lr_decay_step) / decay_steps, 0, 1))
    c = 0.5f0 * (1f0 + cos(Float32(pi) * t))
    lr_min + (lr - lr_min) * c
end

function report_backend!()
    CUDA.functional() || error("CUDA isn't working; this code assumes a GPU.")
    dev = CUDA.device()
    devname = try
        CUDA.name(dev)
    catch
        "?"
    end
    @info "CUDA: $dev ($devname)"
    dev
end
