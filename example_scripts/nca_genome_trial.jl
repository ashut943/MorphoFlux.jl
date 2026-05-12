include(joinpath(@__DIR__, "..", "genome-ae_nca", "genome.jl"))
include(joinpath(@__DIR__, "target_prep.jl"))

using CUDA, LinearAlgebra, Statistics, Plots
using Colors: RGB

emojis = ["🐟", "🐠", "🐡", "🦈", "🐢", "🐊", "🐍", "🦎"]
const N_TARGETS = length(emojis)

hp = GenomeHParams(
    channel_n = 16,
    hidden_n = 128,
    target_size = 40,
    target_padding = 16,
    batch_size = 32,
    pool_size = 256,
    train_steps = 8000,
    min_steps = 64,
    max_steps = 96,
    lr = 1f-3,
    lr_decay_step = 1000,
    lr_decay_factor = 0.01f0,
    encoder_hidden_n = 256,
    hyper_hidden_n = 256,
    kl_beta = 5f-3,
    fire_rate = 0.5f0,
    damage_n = 0,
    use_pattern_pool = true,
    genome_k = 8,
    n_targets = N_TARGETS,
)

run_id = run_timestamp()
run_dir = joinpath("data", run_id)
paths = String[]

for (m, e) in enumerate(emojis)
    d = joinpath(run_dir, "outputs", "prep_$m")
    mkpath(d)
    p = joinpath(run_dir, "target_$m.jld2")
    make_random_emoji_target!(p; target_size=hp.target_size, output_dir=d, run_id, emoji=e)
    # make_png_target!(p, pngs[m]; target_size=hp.target_size, output_dir=d)
    push!(paths, p)
end

model, loss_log = main_genome(
    paths;
    hp,
    data_dir="data",
    run_id,
    snapshot_every=50,
    snapshot_rollout_steps=hp.max_steps,
    video_rollout_steps=2 * hp.max_steps,
    video_fps=30,
    video_fire_rate=0.5f0,
    image_scale=8,
)


