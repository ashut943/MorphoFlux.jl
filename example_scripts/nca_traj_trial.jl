
include(joinpath(@__DIR__, "..", "src", "NeuralCellularAutomata.jl"))
using .NeuralCellularAutomata
using CUDA, FileIO, ImageIO, JLD2
include(joinpath(@__DIR__, "target_prep.jl"))

target_size = 40
blob_dir = joinpath(@__DIR__, "..", "greenblob")

# 1. load your RGBA frames at each observation time as Float32 (W,H,4)
frames = [
    load_target_rgba(joinpath(blob_dir, "1.png"); target_size),
    load_target_rgba(joinpath(blob_dir, "2.png"); target_size),
    load_target_rgba(joinpath(blob_dir, "3.png"); target_size),
    load_target_rgba(joinpath(blob_dir, "4.png"); target_size),
]

# 2. map physical times -> CA step indices (your choice of time scale)
obs_steps = [0, 25, 45, 80]

hp = TrajectoryHParams(
    channel_n   = 16,
    obs_steps   = obs_steps,
    batch_size  = 4,
    train_steps = 3000,
)

run_id = run_timestamp()
run_dir = joinpath("data_traj", run_id)
prep = joinpath(run_dir, "outputs", "prep")
mkpath(prep)

for (i, t_obs) in enumerate(obs_steps)
    src = joinpath(blob_dir, "$i.png")
    isfile(src) && cp(src, joinpath(prep, "target_obs$(lpad(t_obs, 4, '0')).png"); force = true)
end
jldsave(joinpath(prep, "targets.jld2"); obs_steps, target_size)

# 3. build x0: RGBA from frame[1], hidden channels = 0
W, H = size(frames[1], 1), size(frames[1], 2)
x0_single = CUDA.zeros(Float32, W, H, hp.channel_n)
x0_single[:, :, 1:4] .= frames[1]
x0 = repeat(reshape(x0_single, W, H, hp.channel_n, 1), 1, 1, 1, hp.batch_size)

# 4. targets = frames at obs_steps (step 0 is your initial, still supervise it)
targets = frames

model, loss_log = main_trajectory!(
    x0,
    targets;
    hp,
    run_id,
    data_dir = "data_traj",
    snapshot_every = 100,
    video_rollout_steps = 2 * maximum(obs_steps),
    video_fps = 30,
    video_fire_rate = 0.5f0,
    image_scale = 8,
)
