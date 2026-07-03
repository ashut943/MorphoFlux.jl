include(joinpath(@__DIR__, "..", "src", "NeuralCellularAutomata.jl"))
using .NeuralCellularAutomata
include(joinpath(@__DIR__, "target_prep.jl"))

target_size = 40#parse(Int, get(ARGS, 1, "40"))

hp = HParams(
    channel_n = 16,
    hidden_n = 128,
    target_size = target_size,
    target_padding = 16,
    batch_size = 8,
    pool_size = 1024,
    train_steps = 5000,
    min_steps = 64,
    max_steps = 96,
    lr = 2f-3,
    lr_decay_step = 1000,
    lr_decay_factor = 0.05f0,
    fire_rate = 0.5f0,
    damage_n = 0,
    use_pattern_pool = true,
    filters = [:id, :sobel_x, :sobel_y, :laplacian], # subset of: :id, :sobel_x, :sobel_y, :avg, :laplacian
    # vc=4 RGBA (default Grow). vc=2 gray + alpha (living mask on ch2; use `make_png_target_mono2!`).
    # vc=1 = single supervised channel, no life mask (not recommended for masks-on-white).
    visible_channels = 4,
)

seed_offsets = [(0, 0), (-8, 0), (8, 0), (0, -8), (0, 8)]

target_emoji = "🦎"
target_png = "photos/frame_0065_source_0105_black.png"

run_id = run_timestamp()
run_dir = joinpath("data", run_id)
prep = joinpath(run_dir, "outputs", "prep")
mkpath(prep)
jld = joinpath(run_dir, "target.jld2")

# Train on the selected emoji target instead of a PNG file.
make_random_emoji_target!(jld; target_size=hp.target_size, output_dir=prep, run_id, emoji=target_emoji)
# make_png_target!(jld, target_png; target_size=hp.target_size, output_dir=prep)
# make_png_target_mono2!(jld, target_png; target_size=hp.target_size, output_dir=prep)

model, loss_log = main(
    jld;
    hp,
    data_dir="data",
    run_id,
    snapshot_every=50,
    snapshot_rollout_steps=hp.max_steps,
    video_rollout_steps=2 * hp.max_steps,
    video_fps=30,
    video_fire_rate=0.5f0,
    extra_seed_video_offsets=seed_offsets,
    image_scale=8,
)
