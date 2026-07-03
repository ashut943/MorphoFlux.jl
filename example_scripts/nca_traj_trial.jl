using NeuralCellularAutomata, CUDA, FileIO, ImageIO

# 1. load your RGBA frames at each observation time as Float32 (W,H,4)
#    (your own loading logic here)
frames = [load_target_rgba("obs_t0.png"), 
          load_target_rgba("obs_t1.png"),
          load_target_rgba("obs_t2.png")]   # Vector of CuArray{Float32,3}

# 2. map physical times -> CA step indices (your choice of time scale)
obs_steps = [0, 25, 60]   # e.g. t=0 → step 0, t=1.0 → step 25, t=2.4 → step 60

hp = TrajectoryHParams(
    channel_n  = 16,
    obs_steps  = obs_steps,
    batch_size = 4,
    train_steps = 3000,
)

# 3. build x0: RGBA from frame[1], hidden channels = 0
W, H = size(frames[1], 1), size(frames[1], 2)
x0_single = CUDA.zeros(Float32, W, H, hp.channel_n)
x0_single[:, :, 1:4] .= frames[1]            # seed RGBA from data
x0 = repeat(reshape(x0_single, W, H, hp.channel_n, 1), 1, 1, 1, hp.batch_size)

# 4. targets = frames at obs_steps (step 0 is your initial, still supervise it)
targets = frames   # Vector of CuArray{Float32,3}

model, loss_log = train_trajectory!(hp, x0, targets)