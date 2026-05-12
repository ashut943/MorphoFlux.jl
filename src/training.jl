function train!(
    hp::HParams,
    target_path::String;
    checkpoint_dir::String = "checkpoints",
    output_dir::String = "outputs",
    run_id::String = run_timestamp(),
    snapshot_every::Int = 50,
    snapshot_rollout_steps::Int = hp.max_steps,
    video_rollout_steps::Int = 2 * hp.max_steps,
    video_fps::Int = 30,
    image_scale::Int = 8,
    video_fire_rate::Float32 = 1f0,
    extra_seed_video_offsets::Vector{Tuple{Int,Int}} = Tuple{Int,Int}[],
)
    mkpath(checkpoint_dir)
    mkpath(output_dir)
    snapshot_dir = joinpath(output_dir, "training_snapshots")
    @info "run $run_id"

    target = load_target_rgba(target_path; target_size = hp.target_size)
    p = hp.target_padding
    target_padded = CUDA.zeros(Float32,
        hp.target_size + 2p,
        hp.target_size + 2p,
        4,
    )
    target_padded[p+1:p+hp.target_size, p+1:p+hp.target_size, :] .= target

    W, H, _ = size(target_padded)
    seed = make_seed(W, H, hp.channel_n)
    pool = SamplePool(seed, hp.pool_size)

    model = CAModel(hp)
    kernel = build_perception_kernel(hp.channel_n)
    opt_state = Flux.setup(Adam(hp.lr), model)

    loss_log = Float32[]
    train_start = time()
    current_lr = hp.lr

    progress = Progress(hp.train_steps; desc = "Training NCA: ")
    for step_i in 1:hp.train_steps
        iter_start = time()

        current_lr = scheduled_learning_rate(
            step_i, hp.train_steps, hp.lr, hp.lr_decay_step, hp.lr_decay_factor,
        )
        Flux.adjust!(opt_state, current_lr)

        if hp.use_pattern_pool
            batch, idx = sample_batch(pool, hp.batch_size)
            losses = per_sample_loss(batch, target_padded)
            rank = sortperm(losses; rev=true)
            batch = batch[:, :, :, rank]
            batch[:, :, :, 1] .= seed

            if hp.damage_n > 0
                dmasks = make_circle_masks(hp.damage_n, W, H)
                damage = 1f0 .- dmasks
                n_end = hp.batch_size
                n_start = n_end - hp.damage_n + 1
                batch[:, :, :, n_start:n_end] .*= damage
            end
        else
            batch = repeat(reshape(seed, W, H, hp.channel_n, 1), 1, 1, 1, hp.batch_size)
        end

        iter_n = rand(hp.min_steps:hp.max_steps)
        update_masks = make_update_masks(iter_n, W, H, size(batch, 4), hp.fire_rate)

        @info @sprintf("[train] step %d/%d - forward+backward (Flux/Zygote, %d CA steps)...", step_i, hp.train_steps, iter_n)
        CUDA.synchronize()
        fb_start = time()
        (loss_val, x_final), grads = Flux.withgradient(model) do m
            x = batch
            for t in 1:iter_n
                x = step(m, x, kernel; fire_rate = hp.fire_rate, update_mask = @view(update_masks[:, :, :, :, t]))
            end
            return loss_fn(x, target_padded), x
        end
        CUDA.synchronize()
        fb_elapsed = time() - fb_start
        @info @sprintf("[train] step %d - forward+backward done in %.3f s", step_i, fb_elapsed)

        g = grads[1]
        grad_norm = sqrt(sum(Float64(norm(gi))^2 for gi in values(g) if gi !== nothing))
        g_normed = map(g) do gi
            gi === nothing && return nothing
            gi ./ (norm(gi) + 1f-8)
        end

        Flux.update!(opt_state, model, g_normed)
        CUDA.synchronize()

        if hp.use_pattern_pool
            commit!(pool, x_final, idx[rank])
        end

        loss_scalar = Float32(loss_val)
        push!(loss_log, loss_scalar)

        iter_elapsed = time() - iter_start
        total_elapsed = time() - train_start
        @info @sprintf(
            "[train] step %04d/%d | loss %.8e | log10(loss) %.6f | ||g||_2 %.8e | lr %.2e | iter %.3f s | elapsed %.2f s",
            step_i,
            hp.train_steps,
            loss_scalar,
            log10(loss_scalar),
            grad_norm,
            current_lr,
            iter_elapsed,
            total_elapsed,
        )
        next!(
            progress;
            showvalues = [
                (:iteration, step_i),
                (:loss, loss_scalar),
                (:grad_norm, grad_norm),
                (:iter_seconds, iter_elapsed),
                (:elapsed_seconds, total_elapsed),
                (:rollout_steps, iter_n),
            ],
        )

        if snapshot_every > 0 && step_i % snapshot_every == 0
            save_training_snapshot(
                model,
                seed,
                kernel,
                target_padded,
                step_i,
                loss_scalar,
                snapshot_dir;
                run_id,
                n_steps = snapshot_rollout_steps,
                fire_rate = hp.fire_rate,
                scale = image_scale,
            )
        end

        if step_i % 1000 == 0
            fn = joinpath(checkpoint_dir, "model_$(lpad(step_i, 5, '0')).jld2")
            save_model(model, hp, fn)
            @info "Saved checkpoint -> $fn"
        end
    end

    fn = joinpath(checkpoint_dir, "model_final.jld2")
    save_model(model, hp, fn)
    @info "Training complete. Final checkpoint -> $fn"

    save_loss_plots(loss_log, output_dir; run_id)

    save_rollout_video(
        model,
        seed,
        kernel;
        n_steps = video_rollout_steps,
        frame_dir = joinpath(output_dir, "final_rollout_frames"),
        out_path = joinpath(output_dir, "final_rollout_2x.mp4"),
        fps = video_fps,
        fire_rate = video_fire_rate,
        scale = image_scale,
    )

    if !isempty(extra_seed_video_offsets)
        save_multi_seed_videos(
            model,
            W,
            H,
            hp.channel_n,
            kernel;
            seed_offsets = extra_seed_video_offsets,
            n_steps = video_rollout_steps,
            output_dir,
            run_id,
            fps = video_fps,
            fire_rate = video_fire_rate,
            scale = image_scale,
        )
    end

    return model, loss_log
end
