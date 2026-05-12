# target_paths length must match hp.n_targets; entries are .jld2 or .png like load_target_rgba.
function main_genome(
    target_paths::Vector{String};
    hp::GenomeHParams = GenomeHParams(),
    run_id::String = NCA.run_timestamp(),
    data_dir::String = "data",
    snapshot_every::Int = 50,
    snapshot_rollout_steps::Int = hp.max_steps,
    video_rollout_steps::Int = 2 * hp.max_steps,
    video_fps::Int = 30,
    video_fire_rate::Float32 = 1f0,
    image_scale::Int = 8,
)
    NCA.report_backend!()
    M = hp.n_targets
    length(target_paths) == M || error("want $M target paths, got $(length(target_paths))")
    all(isfile, target_paths) || error("some target path missing")

    run_dir = joinpath(data_dir, run_id)
    output_dir = joinpath(run_dir, "outputs")
    checkpoint_dir = joinpath(run_dir, "checkpoints")
    mkpath(run_dir)

    model, loss_log = train_genome!(
        hp, target_paths;
        checkpoint_dir,
        output_dir,
        run_id,
        snapshot_every,
        snapshot_rollout_steps,
        video_rollout_steps,
        video_fps,
        image_scale,
        video_fire_rate,
    )
    @info "genome run finished, loss tail $(loss_log[end])"
    model, loss_log
end
