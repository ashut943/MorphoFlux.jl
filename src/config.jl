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
