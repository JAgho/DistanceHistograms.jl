module DistanceHistograms

using Distances, StaticArrays, TiledIteration
as_ints(a::AbstractArray{CartesianIndex{L}}) where L = reshape(reinterpret(Int, a), (L, size(a)...))

@inline function cross_corr(r1, r2, histo, rmax::Int64, nbins::Int64, metric::Metric=Euclidean())
    rb = nbins/rmax
    @inbounds for i in 1:length(r1)
        @inbounds for j in 1:length(r2)
            d=evaluate(metric, r1[i], r2[j])
            rIndex=floor(Int, d*rb) + 1
            (rIndex < nbins+1) && (histo[rIndex] += 1)
        end
    end
    return histo
end

@generated function _cross_corr_blocked(r1, r2, histo, rmax::Int64, nbins::Int64, blocksize::Type{Val{N}}, metric::Metric=Euclidean()) where N <: Int
    rb = nbins/rmax
    @inbounds for i in 1:blocksize
        @inbounds for j in 1:blocksize
            d=evaluate(metric, r1[i], r2[j])
            rIndex=floor(Int, d*rb) + 1
            (rIndex < nbins+1) && (histo[rIndex] += 1)
        end
    end
    return histo
end

@inline function auto_corr(r, histo, rmax, nbins, metric::Metric=Euclidean())
    rb = (nbins)/rmax 
    @inbounds for i = 1:length(r)
        for j = i+1:length(r)
            d=evaluate(metric, r[i], r[j])
            rIndex=floor(Int, d*rb) +1
            (rIndex < nbins+1) && (histo[rIndex] += 1)
        end
    end
    return histo
end

@inline function unitary_corr(r1, rp, histo, rmax::Int64, nbins::Int64, metric::Metric=Euclidean())
    rb = nbins/rmax
    @inbounds for i in 1:length(r1)
            d=evaluate(metric, r1[i], rp)
            rIndex=floor(Int, d*rb) + 1
            (rIndex < nbins+1) && (histo[rIndex] += 1)
        end
    return histo
end

function auto_corr(points, histo::Array{Int64,1}, rmax::Int64, nbins::Int64; blocksize::Int=64, metric::Metric=Euclidean())
    N = length(points)
    if N <= blocksize
        return auto_corr(points,histo, rmax, nbins, metric)
    end
    tiles, ntile = xtiles(N, blocksize)
    histos = [zeros(Int64, length(histo)) for i in 1:Threads.nthreads()]  
    @sync Threads.@threads for i in tiles
        auto_corr(view(points,i), view(histos, Threads.threadid())[1], rmax, nbins, metric) #
    end
    @sync Threads.@threads for (i,j) in ntile
        cross_corr(view(points,i), view(points,j), view(histos, Threads.threadid())[1], rmax, nbins, metric)
    end 
    sum(histos)
end

function xtiles(N, blocksize)
    chunks = cld(N, blocksize)
    tiles = SplitAxis(1:N, chunks)
    ntile = Tuple{UnitRange{Int64}, UnitRange{Int64}}[]
    for i in 1:length(tiles)
        for j in i+1:length(tiles)
            push!(ntile, (tiles[i], tiles[j]))
        end
    end
    return tiles, ntile
end

export as_ints, auto_corr, cross_corr
# Write your package code here.

end
