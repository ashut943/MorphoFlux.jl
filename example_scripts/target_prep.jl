# Emoji or png -> jld2 (+ rgb preview). Needs NeuralCellularAutomata already in Main.
using Downloads
using FileIO: load
using ImageIO
using JLD2: jldsave
using Random

const N = NeuralCellularAutomata

const NOTO_EMOJI_API_URL = "https://api.github.com/repos/googlefonts/noto-emoji/contents/png/128?ref=main"
const NOTO_EMOJI_CACHE_DIR = "noto_emoji_pngs"

function noto_emoji_download_urls(; cache_dir::String = NOTO_EMOJI_CACHE_DIR)
    mkpath(cache_dir)
    index_path = joinpath(cache_dir, "png128_index.json")
    if !isfile(index_path) || filesize(index_path) == 0
        Downloads.download(NOTO_EMOJI_API_URL, index_path)
    end
    index_text = read(index_path, String)
    rx = r"\"download_url\"\s*:\s*\"([^\"]+/emoji_u[0-9a-f_]+\.png)\""
    urls = String[m.captures[1] for m in eachmatch(rx, index_text)]
    isempty(urls) && error("parsed zero emoji urls from $index_path (delete the json and retry)")
    urls
end

function download_random_noto_emoji!(rng::AbstractRNG; cache_dir::String = NOTO_EMOJI_CACHE_DIR)
    url = rand(rng, noto_emoji_download_urls(; cache_dir))
    path = joinpath(cache_dir, basename(url))
    if !isfile(path) || filesize(path) == 0
        Downloads.download(url, path)
    end
    path
end

function emoji_codepoints(emoji::AbstractString)
    cps = [string(UInt32(c), base=16) for c in emoji if c != '\ufe0f']
    isempty(cps) && error("empty emoji string")
    join(cps, "_")
end

function download_noto_emoji!(emoji::AbstractString; cache_dir::String = NOTO_EMOJI_CACHE_DIR)
    mkpath(cache_dir)
    fn = "emoji_u$(emoji_codepoints(emoji)).png"
    path = joinpath(cache_dir, fn)
    if !isfile(path) || filesize(path) == 0
        Downloads.download(
            "https://raw.githubusercontent.com/googlefonts/noto-emoji/main/png/128/$fn",
            path,
        )
    end
    path
end

function make_random_emoji_target!(
    out_jld::String;
    target_size::Int,
    output_dir::String = "outputs",
    run_id::String = N.run_timestamp(),
    emoji = nothing,
    rng::AbstractRNG = Random.default_rng(),
)
    mkpath(output_dir)
    src = isnothing(emoji) ? download_random_noto_emoji!(rng) : download_noto_emoji!(emoji)
    cp(src, joinpath(output_dir, "target_emoji_source.png"); force=true)

    img = load(src)
    arr = N.image_to_nca_target(img, target_size, target_size)
    jldsave(out_jld; target=arr)

    preview = joinpath(output_dir, "target_emoji_processed.png")
    N.save_rgb_png(preview, N.to_rgb(arr); title="target")

    @info isnothing(emoji) ? "random noto emoji -> $out_jld" : "emoji $emoji -> $out_jld"
    out_jld
end

function make_png_target!(out_jld::String, png_path::String; target_size::Int, output_dir::String="outputs")
    isfile(png_path) || error("no file at $png_path")
    mkpath(output_dir)
    cp(png_path, joinpath(output_dir, "target_source.png"); force=true)

    arr = N.image_to_nca_target(load(png_path), target_size, target_size)
    jldsave(out_jld; target=arr)

    preview = joinpath(output_dir, "target_processed.png")
    N.save_rgb_png(preview, N.to_rgb(arr); title="target")
    @info "png $png_path -> $out_jld"
    out_jld
end
