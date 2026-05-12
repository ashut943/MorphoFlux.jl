function train_genome!(
    hp::GenomeHParams,
    target_paths::Vector{String};
    checkpoint_dir::String = "checkpoints",
    output_dir::String = "outputs",
    run_id::String = NCA.run_timestamp(),
    snapshot_every::Int = 50,
    snapshot_rollout_steps::Int = hp.max_steps,
    video_rollout_steps::Int = 2 * hp.max_steps,
    video_fps::Int = 30,
    image_scale::Int = 8,
    video_fire_rate::Float32 = 1f0,
)
    M = hp.n_targets
    length(target_paths) == M || error("need $M targets, got $(length(target_paths))")
    hp.batch_size % M == 0 || error("batch_size should divide n_targets")

    mkpath(checkpoint_dir)
    mkpath(output_dir)
    snapshot_dir = joinpath(output_dir, "training_snapshots")
    @info "run $run_id  k=$(hp.genome_k)  M=$M"

    p = hp.target_padding
    targets = CuArray{Float32,3}[]
    for (m, path) in enumerate(target_paths)
        t = NCA.load_target_rgba(path; target_size = hp.target_size)
        t_padded = CUDA.zeros(Float32, hp.target_size + 2p, hp.target_size + 2p, 4)
        t_padded[p+1:p+hp.target_size, p+1:p+hp.target_size, :] .= t
        push!(targets, t_padded)
        @info "$m  $path"
    end

    W, H, _ = size(targets[1])
    target_stack = cat([reshape(targets[m], W, H, 4, 1) for m in 1:M]...; dims=4)
    seed  = NCA.make_seed(W, H, hp.channel_n)
    pools = [NCA.SamplePool(seed, hp.pool_size) for _ in 1:M]

    model = GenomeCAModel(hp)
    kernel = NCA.build_perception_kernel(hp.channel_n)
    opt_state = Flux.setup(Adam(hp.lr), model)
    per_m     = div(hp.batch_size, M)   # batch items per target
    target_parts = [repeat(reshape(targets[m], W, H, 4, 1), 1, 1, 1, per_m) for m in 1:M]

    loss_log = Float32[]
    current_lr = hp.lr

    progress = Progress(hp.train_steps; desc="genome train")
    for step_i in 1:hp.train_steps
        iter_start = time()

        current_lr = NCA.scheduled_learning_rate(
            step_i, hp.train_steps, hp.lr, hp.lr_decay_step, hp.lr_decay_factor,
        )
        Flux.adjust!(opt_state, current_lr)

        batch_parts = CuArray{Float32,4}[]
        commit_infos = Tuple{Int,Vector{Int},Vector{Int}}[]

        for m in 1:M
            if hp.use_pattern_pool
                batch_m, idx_m = NCA.sample_batch(pools[m], per_m)
                losses_m = per_sample_loss_single_target(batch_m, targets[m])
                rank_m   = sortperm(losses_m; rev=true)
                batch_m  = batch_m[:, :, :, rank_m]
                batch_m[:, :, :, 1] .= seed   # replace worst with fresh seed
            else
                batch_m = repeat(reshape(seed, W, H, hp.channel_n, 1), 1, 1, 1, per_m)
                idx_m   = collect(1:per_m)
                rank_m  = collect(1:per_m)
            end
            push!(batch_parts, batch_m)
            push!(commit_infos, (m, idx_m, rank_m))
        end

        iter_n       = rand(hp.min_steps:hp.max_steps)
        update_masks = NCA.make_update_masks(iter_n, W, H, hp.batch_size, hp.fire_rate)
        latent_eps   = CUDA.randn(Float32, hp.genome_k, M)

        CUDA.synchronize()
        fb_start = time()
        (loss_val, x_final, recon_val, kl_val), grads = Flux.withgradient(model) do m
            mu, logvar = encode_targets(m, target_stack)
            g_batch = sample_latents(mu, logvar, latent_eps)
            theta_batch = hypernetwork_theta(m, g_batch)
            kl_loss = latent_kl_loss(mu, logvar)

            x_final_parts = map(1:M) do target_i
                theta = unpack_generated_theta(m, theta_batch[:, target_i:target_i])
                x = batch_parts[target_i]
                b_start = (target_i - 1) * per_m + 1
                b_end = target_i * per_m
                for t in 1:iter_n
                    x = step_generated_nca(m, x, kernel, theta;
                        fire_rate   = hp.fire_rate,
                        update_mask = @view(update_masks[:, :, :, b_start:b_end, t]),
                    )
                end
                x
            end

            recon_loss = genome_loss_fn(x_final_parts[1], target_parts[1])
            for target_i in 2:M
                recon_loss = recon_loss + genome_loss_fn(x_final_parts[target_i], target_parts[target_i])
            end
            recon_loss = recon_loss / Float32(M)
            total_loss = recon_loss + hp.kl_beta * kl_loss
            return total_loss, cat(x_final_parts...; dims=4), recon_loss, kl_loss
        end
        CUDA.synchronize()
        fb_elapsed = time() - fb_start
        @info @sprintf("step %d fwd+bwd %.2fs", step_i, fb_elapsed)

        isfinite(Float32(loss_val)) || error("Non-finite loss at step $step_i: loss=$loss_val recon=$recon_val kl=$kl_val")

        g = grads[1]
        for (name, gi) in pairs(g)
            gi === nothing && continue
            finite_grad = gi isa Number ? isfinite(gi) : all(isfinite, Array(gi))
            finite_grad || error("Non-finite gradient for $name at step $step_i")
        end
        grad_norm = sqrt(sum(Float64(norm(gi))^2 for gi in values(g) if gi !== nothing))
        g_normed  = map(g) do gi
            gi === nothing && return nothing
            gi ./ (norm(gi) + 1f-8)
        end
        Flux.update!(opt_state, model, g_normed)
        CUDA.synchronize()

        if hp.use_pattern_pool
            for (i, (m, idx_m, rank_m)) in enumerate(commit_infos)
                b_start = (i - 1) * per_m + 1
                b_end   = i * per_m
                NCA.commit!(pools[m], x_final[:, :, :, b_start:b_end], idx_m[rank_m])
            end
        end

        loss_scalar   = Float32(loss_val)
        push!(loss_log, loss_scalar)
        iter_elapsed = time() - iter_start

        @info @sprintf(
            "%d/%d loss %.3e recon %.3e kl %.3e |g| %.2e lr %.1e %.1fs",
            step_i, hp.train_steps, loss_scalar, Float32(recon_val), Float32(kl_val),
            grad_norm, current_lr, iter_elapsed,
        )
        next!(progress; showvalues=[
            (:step, step_i), (:loss, loss_scalar), (:recon, Float32(recon_val)),
            (:kl, Float32(kl_val)), (:gnorm, grad_norm), (:s, iter_elapsed), (:T, iter_n),
        ])

        batch_parts = nothing
        update_masks = nothing
        latent_eps = nothing
        x_final = nothing
        grads = nothing
        g = nothing
        g_normed = nothing
        if step_i % 10 == 0
            GC.gc()
            CUDA.reclaim()
        end

        if snapshot_every > 0 && step_i % snapshot_every == 0
            save_genome_training_snapshot(
                model, seed, kernel, targets, step_i, loss_scalar, snapshot_dir;
                run_id,
                n_steps = snapshot_rollout_steps,
                scale   = image_scale,
            )
        end

        if step_i % 1000 == 0
            fn = joinpath(checkpoint_dir, "genome_model_$(lpad(step_i, 5, '0')).jld2")
            save_genome_model(model, hp, fn)
            @info "ckpt $fn"
        end
    end

    fn = joinpath(checkpoint_dir, "genome_model_final.jld2")
    save_genome_model(model, hp, fn)
    @info "wrote $fn"

    NCA.save_loss_plots(loss_log, output_dir; run_id)

    for m in 1:M
        save_genome_rollout_video(
            model, seed, kernel, targets[m], m;
            n_steps   = video_rollout_steps,
            frame_dir = joinpath(output_dir, "target_$(m)_frames"),
            out_path  = joinpath(output_dir, "target_$(m)_rollout.mp4"),
            fps       = video_fps,
            fire_rate = video_fire_rate,
            scale     = image_scale,
        )
    end

    return model, loss_log
end
