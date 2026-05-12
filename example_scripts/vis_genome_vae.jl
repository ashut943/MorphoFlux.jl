#PCA and MDS plots, for fun
include(joinpath(@__DIR__, "..", "genome-ae_nca", "genome.jl"))

using CUDA, LinearAlgebra, Statistics, Plots
using Colors: RGB

run_dir = "data/20260503_100849"
model, hp = load_genome_model(joinpath(run_dir, "checkpoints", "genome_model_final.jld2"))

emojis = ["🐟", "🐠", "🐡", "🦈", "🐢", "🐊", "🐍", "🦎"]

p = hp.target_padding
W = hp.target_size + 2p

targets = CuArray{Float32,3}[]
target_thumbs = Array{Float32,3}[]
for m in 1:hp.n_targets
    t = load_target_rgba(joinpath(run_dir, "target_$(m).jld2"); target_size=hp.target_size)
    push!(target_thumbs, NCA.to_rgb(Array(t)))
    padded = CUDA.zeros(Float32, W, W, 4)
    padded[p+1:p+hp.target_size, p+1:p+hp.target_size, :] .= t
    push!(targets, padded)
end

target_stack = cat([reshape(targets[m], W, W, 4, 1) for m in 1:hp.n_targets]...; dims=4)

mu, _ = encode_targets(model, target_stack)

Z = Array(mu)'
Zc = Z .- mean(Z; dims=1)

F = svd(Zc)
P = Zc * F.V[:, 1:2]
P3 = Zc * F.V[:, 1:3]

function classical_mds(Z::AbstractMatrix{<:Real}, dims::Int)
    n = size(Z, 1)
    D2 = zeros(Float64, n, n)
    for i in 1:n, j in 1:n
        D2[i, j] = sum((Float64.(Z[i, :]) .- Float64.(Z[j, :])) .^ 2)
    end
    J = I - fill(1 / n, n, n)
    B = -0.5 .* J * D2 * J
    eig = eigen(Symmetric(B))
    order = sortperm(eig.values; rev=true)
    keep = order[1:min(dims, length(order))]
    vals = max.(eig.values[keep], 0.0)
    return eig.vectors[:, keep] * Diagonal(sqrt.(vals))
end

M2 = classical_mds(Z, 2)
M3 = classical_mds(Z, 3)

mkpath(joinpath(run_dir, "outputs"))

plt_pca = scatter(P[:, 1], P[:, 2]; label=false)
for i in eachindex(emojis)
    annotate!(plt_pca, P[i, 1], P[i, 2], text(emojis[i], 14))
end
savefig(plt_pca, joinpath(run_dir, "outputs", "genome_pca.png"))

function scale_nearest(img::AbstractArray{<:Real,3}, scale::Int)
    W, H, C = size(img)
    out = ones(Float32, W * scale, H * scale, C)
    for x in 1:W, y in 1:H, c in 1:C
        out[(x-1)*scale+1:x*scale, (y-1)*scale+1:y*scale, c] .= Float32(img[x, y, c])
    end
    return out
end

function paste_thumbnail!(canvas, thumb, cx::Int, cy::Int)
    W, H, _ = size(canvas)
    tw, th, _ = size(thumb)
    x0 = clamp(cx - tw ÷ 2, 1, W - tw + 1)
    y0 = clamp(cy - th ÷ 2, 1, H - th + 1)

    border = 2
    bx0 = max(1, x0 - border)
    by0 = max(1, y0 - border)
    bx1 = min(W, x0 + tw + border - 1)
    by1 = min(H, y0 + th + border - 1)
    canvas[bx0:bx1, by0:by1, :] .= 0.15f0
    canvas[x0:x0+tw-1, y0:y0+th-1, :] .= thumb
    return nothing
end

function save_thumbnail_embedding(path::String, P, target_thumbs; title::String, canvas_w::Int=900, canvas_h::Int=620)
    canvas = ones(Float32, canvas_w, canvas_h, 3)
    margin = 90
    xs = P[:, 1]
    ys = P[:, 2]
    xrange = maximum(xs) - minimum(xs)
    yrange = maximum(ys) - minimum(ys)
    xrange = xrange == 0 ? 1 : xrange
    yrange = yrange == 0 ? 1 : yrange

    function project_point(x, y)
        px = margin + round(Int, (x - minimum(xs)) / xrange * (canvas_w - 2margin))
        py = canvas_h - margin - round(Int, (y - minimum(ys)) / yrange * (canvas_h - 2margin))
        return px, py
    end

    for frac in 0.25:0.25:0.75
        gx = margin + round(Int, frac * (canvas_w - 2margin))
        gy = margin + round(Int, frac * (canvas_h - 2margin))
        canvas[gx:gx, margin:canvas_h-margin, :] .= 0.9f0
        canvas[margin:canvas_w-margin, gy:gy, :] .= 0.9f0
    end

    for i in axes(P, 1)
        px, py = project_point(P[i, 1], P[i, 2])
        thumb = scale_nearest(target_thumbs[i], 2)
        paste_thumbnail!(canvas, thumb, px, py)
    end

    NCA.save_rgb_png(path, canvas; title, fig_size=(canvas_w, canvas_h))
end

save_thumbnail_embedding(
    joinpath(run_dir, "outputs", "genome_pca_thumbnails.png"),
    P,
    target_thumbs;
    title="Genome PCA: target thumbnails at encoder mean locations",
)
save_thumbnail_embedding(
    joinpath(run_dir, "outputs", "genome_pca_thumbnails.pdf"),
    P,
    target_thumbs;
    title="Genome PCA-2D: target thumbnails at encoder mean locations",
)

save_thumbnail_embedding(
    joinpath(run_dir, "outputs", "genome_mds2_thumbnails.png"),
    M2,
    target_thumbs;
    title="Genome MDS-2D: pairwise distances between encoder means",
)
save_thumbnail_embedding(
    joinpath(run_dir, "outputs", "genome_mds2_thumbnails.pdf"),
    M2,
    target_thumbs;
    title="Genome MDS-2D: pairwise distances between encoder means",
)

point_colors = [
    :dodgerblue,
    :orange,
    :gold,
    :gray45,
    :forestgreen,
    :saddlebrown,
    :olivedrab,
    :limegreen,
]

function rgb_matrix(canvas::AbstractArray{<:Real,3})
    img = permutedims(clamp.(Float32.(canvas), 0f0, 1f0), (2, 1, 3))
    mat = Matrix{RGB{Float64}}(undef, size(img, 1), size(img, 2))
    for y in axes(img, 1), x in axes(img, 2)
        mat[y, x] = RGB(Float64(img[y, x, 1]), Float64(img[y, x, 2]), Float64(img[y, x, 3]))
    end
    return mat
end

function thumbnail_legend_plot(target_thumbs, point_colors)
    row_h = 72
    canvas_w = 220
    canvas_h = row_h * length(target_thumbs) + 20
    canvas = ones(Float32, canvas_w, canvas_h, 3)

    for i in eachindex(target_thumbs)
        y = 12 + (i - 1) * row_h
        thumb = scale_nearest(target_thumbs[i], 1)
        paste_thumbnail!(canvas, thumb, 95, y + row_h ÷ 2)
    end

    plt = plot(
        rgb_matrix(canvas);
        title="Legend",
        showaxis=false,
        grid=false,
        ticks=false,
        aspect_ratio=1,
    )
    for i in eachindex(target_thumbs)
        y = 12 + (i - 1) * row_h + row_h ÷ 2
        annotate!(plt, 35, y, text(string(i), 12, point_colors[i]))
    end
    return plt
end

function save_3d_embedding(name::String, coords, target_thumbs, point_colors; method::String)
    embedding = plot(
        xlabel="$method 1",
        ylabel="$method 2",
        zlabel="$method 3",
        title="Genome $method-3D: encoder mean coordinates",
    )
    for i in axes(coords, 1)
        scatter!(
            embedding,
            [coords[i, 1]], [coords[i, 2]], [coords[i, 3]];
            label=string(i),
            marker=(:circle, 7, point_colors[i]),
        )
        annotate!(embedding, coords[i, 1], coords[i, 2], coords[i, 3], text(string(i), 10, point_colors[i]))
    end

    savefig(embedding, joinpath(run_dir, "outputs", "$(name).png"))
    savefig(embedding, joinpath(run_dir, "outputs", "$(name).pdf"))

    with_legend = plot(
        embedding,
        thumbnail_legend_plot(target_thumbs, point_colors);
        layout=@layout([a{0.74w} b]),
        size=(1200, 680),
    )
    savefig(with_legend, joinpath(run_dir, "outputs", "$(name)_with_legend.png"))
    savefig(with_legend, joinpath(run_dir, "outputs", "$(name)_with_legend.pdf"))
    return embedding, with_legend
end

save_3d_embedding("genome_pca3", P3, target_thumbs, point_colors; method="PCA")
save_3d_embedding("genome_mds3", M3, target_thumbs, point_colors; method="MDS")

open(joinpath(run_dir, "outputs", "genome_embedding_labels.txt"), "w") do io
    for i in eachindex(emojis)
        println(io, "$i => $(emojis[i])")
    end
end
