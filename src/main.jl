function default_hparams()
    HParams(
        channel_n = 16,
        hidden_n = 128,
        target_size = 40,
        target_padding = 16,
        batch_size = 8,
        pool_size = 1024,
        train_steps = 8000,
        fire_rate = 0.5f0,
        damage_n = 3,
        use_pattern_pool = true,
    )
end

# target_path: .jld2 with `target`, or .png (see load_target_rgba)
function main(
    target_path::String;
    hp::HParams = default_hparams(),
    run_id::String = run_timestamp(),
    data_dir::String = "data",
    snapshot_every::Int = 50,
    snapshot_rollout_steps::Int = hp.max_steps,
    video_rollout_steps::Int = 2 * hp.max_steps,
    video_fps::Int = 30,
    image_scale::Int = 8,
    video_fire_rate::Float32 = 1f0,
    extra_seed_video_offsets::Vector{Tuple{Int,Int}} = Tuple{Int,Int}[],
)
    report_backend!()
    isfile(target_path) || error("bad target_path: $target_path")

    run_dir = joinpath(data_dir, run_id)
    output_dir = joinpath(run_dir, "outputs")
    checkpoint_dir = joinpath(run_dir, "checkpoints")
    mkpath(run_dir)

    model, loss_log = train!(
        hp,
        target_path;
        checkpoint_dir,
        output_dir,
        run_id,
        snapshot_every,
        snapshot_rollout_steps,
        video_rollout_steps,
        video_fps,
        image_scale,
        video_fire_rate,
        extra_seed_video_offsets,
    )
    @info "done, final loss $(loss_log[end])"
    model, loss_log
end

function main_trajectory!(
    x0::CuArray{Float32,4},
    targets::Vector{<:CuArray{Float32,3}};
    hp::TrajectoryHParams = TrajectoryHParams(),
    run_id::String = run_timestamp(),
    data_dir::String = "data_traj",
    snapshot_every::Int = 100,
    video_rollout_steps::Int = 0,
    video_fps::Int = 30,
    image_scale::Int = 8,
    video_fire_rate::Float32 = 1f0,
)
    report_backend!()
    length(targets) == length(hp.obs_steps) ||
        error("length(targets) ($(length(targets))) must match length(hp.obs_steps) ($(length(hp.obs_steps)))")

    run_dir = joinpath(data_dir, run_id)
    output_dir = joinpath(run_dir, "outputs")
    checkpoint_dir = joinpath(run_dir, "checkpoints")
    mkpath(run_dir)

    model, loss_log = train_trajectory!(
        hp,
        x0,
        targets;
        checkpoint_dir,
        output_dir,
        run_id,
        snapshot_every,
        video_rollout_steps,
        video_fps,
        image_scale,
        video_fire_rate,
    )
    @info "done, final loss $(loss_log[end])"
    model, loss_log
end
