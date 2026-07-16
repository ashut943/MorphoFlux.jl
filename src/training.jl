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
    seed = make_seed(W, H, hp.channel_n; visible_channels = hp.visible_channels)
    pool = SamplePool(seed, hp.pool_size)

    model = CAModel(hp)
    kernel = build_perception_kernel(hp.channel_n; filters = hp.filters)
    opt_state = Flux.setup(AdamW(hp.lr, (0.9f0, 0.999f0), 1f-3), model)

    update_mask_buf = CUDA.zeros(Float32, W, H, 1, hp.batch_size, hp.max_steps)
    batch_buf = CUDA.zeros(Float32, W, H, hp.channel_n, hp.batch_size)

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
            idx = randperm(pool.size)[1:hp.batch_size]
            sample_batch!(batch_buf, pool, idx)
            losses = per_sample_loss(batch_buf, target_padded; vc = hp.visible_channels)
            rank = sortperm(losses; rev=true)
            sample_batch!(batch_buf, pool, idx[rank])
            batch_buf[:, :, :, 1] .= seed

            if hp.damage_n > 0
                dmasks = make_circle_masks(hp.damage_n, W, H)
                damage = 1f0 .- dmasks
                n_end = hp.batch_size
                n_start = n_end - hp.damage_n + 1
                batch_buf[:, :, :, n_start:n_end] .*= damage
            end
        else
            batch_buf .= reshape(seed, W, H, hp.channel_n, 1)
        end

        iter_n = rand(hp.min_steps:hp.max_steps)
        update_masks = fill_update_masks!(update_mask_buf, iter_n, hp.fire_rate)

        (loss_val, x_final), grads = Flux.withgradient(model) do m
            x = batch_buf
            for t in 1:iter_n
                x = step(m, x, kernel; fire_rate = hp.fire_rate, update_mask = @view(update_masks[:, :, :, :, t]))
            end
            return loss_fn(x, target_padded; vc = hp.visible_channels), x
        end

        g = grads[1]
        grad_norm = sqrt(sum(Float64(norm(gi))^2 for gi in values(g) if gi !== nothing))
        for gi in values(g)
            gi === nothing && continue
            gi ./= (norm(gi) + 1f-8)
        end

        Flux.update!(opt_state, model, g)

        if hp.use_pattern_pool
            commit!(pool, x_final, idx[rank])
        end

        if step_i % 50 == 0
            GC.gc(false)
            CUDA.reclaim()
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
                vc = hp.visible_channels,
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


function train_trajectory!(
    hp::TrajectoryHParams,
    x0::CuArray{Float32,4},             
    targets::Vector{<:CuArray{Float32,3}},
    weights::Vector{Float32} = ones(Float32, length(targets));
    checkpoint_dir::String = "checkpoints",
    output_dir::String     = "outputs",
    run_id::String         = run_timestamp(),
    snapshot_every::Int    = 100,
    video_rollout_steps::Int = 0,
    video_fps::Int = 30,
    image_scale::Int = 8,
    video_fire_rate::Float32 = 1f0,
)
    mkpath(checkpoint_dir); mkpath(output_dir)
    snapshot_dir = joinpath(output_dir, "training_snapshots")
    @assert length(targets) == length(hp.obs_steps)

    model  = CAModel(hp)
    kernel = build_perception_kernel(hp.channel_n; filters = hp.filters)
    opt_state = Flux.setup(AdamW(hp.lr, (0.9f0, 0.999f0), 1f-3), model)

    K = maximum(hp.obs_steps)    # total steps to roll out each iteration
    rollout_steps = video_rollout_steps > 0 ? video_rollout_steps : 2 * K

    W, H, N = size(x0, 1), size(x0, 2), size(x0, 4)
    update_mask_buf = CUDA.zeros(Float32, W, H, 1, N, K)

    loss_log = Float32[]
    train_start = time()

    progress = Progress(hp.train_steps; desc = "Trajectory fitting: ")
    for step_i in 1:hp.train_steps

        current_lr = scheduled_learning_rate(
            step_i, hp.train_steps, hp.lr, hp.lr_decay_step, hp.lr_decay_factor,
        )
        Flux.adjust!(opt_state, current_lr)

        update_masks = fill_update_masks!(update_mask_buf, K, hp.fire_rate)

        loss_val, grads = Flux.withgradient(model) do m
            x = x0
            total_loss = 0f0
            t_prev = 0
            for (i, t_obs) in enumerate(hp.obs_steps)
                for t in (t_prev + 1):t_obs
                    x = step(m, x, kernel; fire_rate = hp.fire_rate, update_mask = @view(update_masks[:, :, :, :, t]))
                end
                total_loss += loss_fn(x, targets[i]; vc = hp.visible_channels)
                t_prev = t_obs
            end
            return total_loss / length(hp.obs_steps)
        end

        g = grads[1]
        grad_norm = sqrt(sum(Float64(norm(gi))^2 for gi in values(g) if gi !== nothing))
        for gi in values(g)
            gi === nothing && continue
            gi ./= (norm(gi) + 1f-8)
        end

        Flux.update!(opt_state, model, g)
        if step_i % 50 == 0
            GC.gc(false)
            CUDA.reclaim()
        end

        loss_scalar = Float32(loss_val)
        push!(loss_log, loss_scalar)

        @info @sprintf(
            "[traj] step %04d/%d | loss %.6e | log10 %.4f | ||g|| %.4e | lr %.2e",
            step_i, hp.train_steps, loss_scalar, log10(loss_scalar), grad_norm, current_lr,
        )
        next!(progress; showvalues = [(:loss, loss_scalar), (:grad_norm, grad_norm)])

        if snapshot_every > 0 && step_i % snapshot_every == 0
            fn = joinpath(checkpoint_dir, "traj_$(lpad(step_i,5,'0')).jld2")
            save_model(model, hp, fn)
            save_trajectory_snapshot(
                model,
                x0,
                kernel,
                targets,
                hp.obs_steps,
                step_i,
                loss_scalar,
                snapshot_dir;
                fire_rate = hp.fire_rate,
                vc = hp.visible_channels,
            )
        end
    end

    save_model(model, hp, joinpath(checkpoint_dir, "traj_final.jld2"))
    save_loss_plots(loss_log, output_dir; run_id)

    seed = x0[:, :, :, 1]
    save_rollout_video(
        model,
        seed,
        kernel;
        n_steps = rollout_steps,
        frame_dir = joinpath(output_dir, "final_rollout_frames"),
        out_path = joinpath(output_dir, "final_rollout_2x.mp4"),
        fps = video_fps,
        fire_rate = video_fire_rate,
        scale = image_scale,
        vc = hp.visible_channels,
    )
    save_trajectory_rollout_video(
        model,
        seed,
        kernel,
        hp.obs_steps,
        targets;
        n_steps = K,
        frame_dir = joinpath(output_dir, "trajectory_comparison_frames"),
        out_path = joinpath(output_dir, "trajectory_comparison.mp4"),
        fps = video_fps,
        fire_rate = video_fire_rate,
        scale = image_scale,
        vc = hp.visible_channels,
    )

    return model, loss_log
end