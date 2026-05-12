function rollout_state_genome(
    model::GenomeCAModel,
    seed::CuArray{Float32,3},
    kernel::CuArray{Float32,4},
    target::CuArray{Float32,3},
    n_steps::Int;
    fire_rate::Float32 = 1f0,
)
    x = reshape(seed, size(seed)..., 1)
    target_batch = reshape(target, size(target, 1), size(target, 2), 4, 1)
    mu, _ = encode_targets(model, target_batch)
    theta = unpack_generated_theta(model, hypernetwork_theta(model, mu))
    for _ in 1:n_steps
        x = step_generated_nca(model, x, kernel, theta; fire_rate)
    end
    return x
end

function save_genome_training_snapshot(
    model::GenomeCAModel,
    seed::CuArray{Float32,3},
    kernel::CuArray{Float32,4},
    targets::Vector{<:CuArray{Float32,3}},
    step_i::Int,
    loss_val::Real,
    snapshot_dir::String;
    run_id::String = NCA.run_timestamp(),
    n_steps::Int,
    fire_rate::Float32 = 1f0,
    scale::Int = 8,
    n_cols_pairs::Int = 8,   
)
    M        = length(targets)
    pair_gap = 2   
    col_gap  = 6   
    row_gap  = 6   
    n_rows   = cld(M, n_cols_pairs)

    panels = Array{Float32,3}[]
    for m in 1:M
        x = rollout_state_genome(model, seed, kernel, targets[m], n_steps; fire_rate)
        push!(panels, NCA.to_rgb(Array(x)[:, :, :, 1]))
        push!(panels, NCA.to_rgb(Array(targets[m])))
    end

    W, H, _ = size(panels[1])
    pair_w   = 2W + pair_gap
    canvas_W = n_cols_pairs * pair_w + (n_cols_pairs - 1) * col_gap
    canvas_H = n_rows * H + (n_rows - 1) * row_gap

    canvas = ones(Float32, canvas_W, canvas_H, 3)
    for m in 1:M
        row = div(m - 1, n_cols_pairs) 
        col = (m - 1) % n_cols_pairs   
   
        x0  = col * (pair_w + col_gap) + 1
        y0  = row * (H + row_gap) + 1
        canvas[x0:x0+W-1, y0:y0+H-1, :]           .= Float32.(panels[2m-1])
        canvas[x0+W+pair_gap:x0+pair_w-1, y0:y0+H-1, :] .= Float32.(panels[2m])
    end

    fig_w = clamp(canvas_W * 4, 1200, 8000)
    fig_h = round(Int, fig_w * canvas_H / canvas_W) + 60   

    mkpath(snapshot_dir)
    path = joinpath(snapshot_dir, @sprintf("train_step_%05d.png", step_i))
    NCA.save_rgb_png(path, canvas;
        title    = @sprintf("step %05d | loss %.4g | K=%d M=%d",
                             step_i, loss_val,
                             model.latent_k, M),
        fig_size = (fig_w, fig_h))
    @info "saved $path ($(round(loss_val; sigdigits=3)))"
    return path
end

function save_genome_rollout_video(
    model::GenomeCAModel,
    seed::CuArray{Float32,3},
    kernel::CuArray{Float32,4},
    target::CuArray{Float32,3},
    target_idx::Int;
    n_steps::Int,
    frame_dir::String,
    out_path::String,
    fps::Int = 30,
    fire_rate::Float32 = 1f0,
    scale::Int = 8,
)
    rm(frame_dir; force=true, recursive=true)
    mkpath(frame_dir)

    x = reshape(seed, size(seed)..., 1)
    target_batch = reshape(target, size(target, 1), size(target, 2), 4, 1)
    mu, _ = encode_targets(model, target_batch)
    theta = unpack_generated_theta(model, hypernetwork_theta(model, mu))

    progress = Progress(n_steps + 1; desc="rollout #$target_idx ")
    for i in 0:n_steps
        rgb = NCA.to_rgb(Array(x)[:, :, :, 1])
        NCA.write_rgb_ppm(joinpath(frame_dir, @sprintf("frame_%05d.ppm", i)), rgb; scale)
        next!(progress; showvalues = [(:frame, i), (:target, target_idx)])
        if i < n_steps
            x = step_generated_nca(model, x, kernel, theta; fire_rate)
        end
    end

    ffmpeg = FFMPEG_jll.ffmpeg()
    run(`$ffmpeg -y -framerate $fps -i $(joinpath(frame_dir, "frame_%05d.ppm")) -pix_fmt yuv420p $out_path`)
    @info "video -> $out_path"
    return out_path
end

function run_ca_genome(
    model::GenomeCAModel,
    seed::CuArray{Float32,3},
    kernel::CuArray{Float32,4},
    target::CuArray{Float32,3},
    n_steps::Int;
    fire_rate::Float32 = model.fire_rate,
)
    x = rollout_state_genome(model, seed, kernel, target, n_steps; fire_rate)
    Array(x)
end
