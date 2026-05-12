mutable struct SamplePool
    x::CuArray{Float32,4}
    size::Int
end

function SamplePool(seed::CuArray{Float32,3}, pool_size::Int)
    pool = repeat(reshape(seed, size(seed)..., 1), 1, 1, 1, pool_size)
    return SamplePool(pool, pool_size)
end

function sample_batch(pool::SamplePool, batch_size::Int)
    idx = randperm(pool.size)[1:batch_size]
    batch = pool.x[:, :, :, idx]
    return batch, idx
end

function commit!(pool::SamplePool, batch::CuArray{Float32,4}, idx::Vector{Int})
    pool.x[:, :, :, idx] .= batch
    return nothing
end
