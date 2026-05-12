using Flux
using Flux: Adam
using CUDA, cuDNN, NNlib, Zygote, Random, Statistics, LinearAlgebra
using JLD2
using ProgressMeter
using Printf
using Dates
using FFMPEG_jll
using FileIO: load
using ImageIO
using Colors: RGB, alpha, blue, green, red

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using Plots
gr()

include(abspath(joinpath(@__DIR__, "..", "src", "NeuralCellularAutomata.jl")))
using .NeuralCellularAutomata
const NCA = NeuralCellularAutomata

include(joinpath(@__DIR__, "genome_config.jl"))
include(joinpath(@__DIR__, "genome_model.jl"))
include(joinpath(@__DIR__, "genome_visualization.jl"))
include(joinpath(@__DIR__, "genome_checkpointing.jl"))
include(joinpath(@__DIR__, "genome_training.jl"))
include(joinpath(@__DIR__, "genome_main.jl"))
