module DistanceHistograms

using Distances, StaticArrays, TiledIteration
as_ints(a::AbstractArray{CartesianIndex{L}}) where L = reshape(reinterpret(Int, a), (L, size(a)...))

function svectorscopy(x::Matrix{T}, ::Val{N}) where {T,N}
    size(x,1) == N || error("sizes mismatch")
    isbitstype(T) || error("use for bitstypes only")
    copy(reinterpret(SVector{N,T}, vec(x)))
end

import Distances.eval_op

#significant performance penalty if you want to bound points beyond the second wigner seitz cell
@inline function eval_op(::PeriodicEuclidean, ai, bi, p) 
    s1 = abs(ai - bi)
    abs2(min(s1, p - s1))
end

"""
    cross_corr(r1::AbstractVector{T}, r2::AbstractVector{T}, rmax::Int, nbins::Int; metric::Metric=Euclidean()) where {T<:SVector}

Compute the cross correlation distance histogram between a a collection of points `r1` and another collection `r2`
"""
function cross_corr(r1::AbstractVector{T}, r2::AbstractVector{T}, rmax::Int64, nbins::Int64; blocksize=1000, metric::Metric=Euclidean()) where {T<:SVector}
    cross_corr!(r1, r2, zeros(Int64, nbins), rmax, nbins; blocksize=blocksize, metric=metric) 
end

@inline function _cross_corr!(r1::AbstractVector{T}, r2::AbstractVector{T}, histo::AbstractVector{Int}, rmax::Int64, nbins::Int64; metric::Metric=Euclidean()) where {T<:SVector}
    rb = nbins/rmax
    @inbounds for i in r1#1:length(r1)
        @inbounds for j in r2#1:length(r2)
            d=evaluate(metric, i, j)
            rIndex=floor(Int64, d*rb) + 1
            @inbounds (rIndex < nbins+1) && (histo[rIndex] += 1)
        end
    end
    return histo
end


"""
    cross_corr(r1::AbstractVector{T}, r2::AbstractVector{T}, histo::AbstractVector{Int}, rmax::Int, nbins::Int; metric::Metric=Euclidean()) where {T<:SVector}

Compute the cross correlation distance histogram between a a collection of points `r1` and another collection `r2` for a preallocated array
"""
function cross_corr!(r1::AbstractVector{T}, r2::AbstractVector{T}, histo::AbstractVector{Int}, rmax::Int64, nbins::Int64; blocksize=1000, metric::Metric=Euclidean()) where {T<:SVector}
    rb = nbins/rmax
    if length(r1)*length(r2) < blocksize*blocksize
        return _cross_corr!(r2, r2, histo, rmax, nbins; metric=metric) 
    end
    histos = [zeros(Int64, nbins) for i in 1:Threads.nthreads()]  
    @inbounds Threads.@threads for i in r1#1:length(r1)
        @inbounds for j in r2#1:length(r2)
            d=evaluate(metric, i, j)
            rIndex=floor(Int64, d*rb) + 1
            @inbounds (rIndex < nbins+1) && (view(histos, Threads.threadid())[1][rIndex] += 1)
        end
    end
    return sum(histos)
end



"""
    unitary_corr(r1::AbstractVector{T}, rp, rmax::Int64, nbins::Int64, metric::Metric=Euclidean()) where {T<:SVector}

Compute the cross correlation distance histogram between a single SVector `rp` and a collection of points `r1` 
"""
function unitary_corr(r1::AbstractVector{T}, rp::T, rmax::Int, nbins::Int; metric::Metric=Euclidean()) where {T<:SVector}
    unitary_corr!(r1, rp, zeros(Int64, nbins), rmax, nbins; metric=metric) 
end

"""
    unitary_corr(r1::AbstractVector{T}, rp, histo, rmax::Int64, nbins::Int64, metric::Metric=Euclidean()) where {T<:SVector}

Compute the cross correlation distance histogram between a single SVector `rp` and a collection of points `r1` and store the result to a preallocated array 
"""
@inline function unitary_corr!(r1::AbstractVector{T}, rp::T, histo::AbstractVector{Int}, rmax::Int, nbins::Int; metric::Metric=Euclidean()) where {T<:SVector}
    rb = nbins/rmax
    @inbounds for i in r1#1:length(r1)
            d=evaluate(metric, i, rp)
            rIndex=floor(Int, d*rb) + 1
            @inbounds (rIndex < nbins+1) && (histo[rIndex] += 1)
        end
    return histo
end

"""
    auto_corr(points::AbstractVector{T}, histo::Array{Int64,1}, rmax::Int64, nbins::Int64; blocksize::Int=64, metric::Metric=Euclidean())) where {T<:SVector}

Compute the autocorrelation distance histogram for a set of SVector points

Threaded implementation for larger autocorrelations
"""
function auto_corr(points::AbstractVector{T}, rmax::Int, nbins::Int; blocksize::Int=1000, metric::Metric=Euclidean()) where {T<:SVector}
    auto_corr!(points, zeros(Int64, nbins), rmax, nbins; blocksize=blocksize, metric=metric) 
end

@inline function _auto_corr!(r::AbstractVector{T}, histo::AbstractVector{Int}, rmax::Int, nbins::Int, metric::Metric=Euclidean()) where {T<:SVector}
    rb = (nbins)/rmax 
    @inbounds for (i, ri) = enumerate(r)#1:length(r)
        for j = i+1:length(r)
            d=evaluate(metric, ri, r[j])
            rIndex=floor(Int, d*rb) +1
            @inbounds (rIndex < nbins+1) && (histo[rIndex] += 1)
        end
    end
    return histo
end

"""
    auto_corr!(points::AbstractVector{T}, histo::AbstractVector{Int}, rmax::Int, nbins::Int; blocksize::Int=64, metric::Metric=Euclidean()) where {T<:SVector}

Compute the autocorrelation distance histogram for a set of SVector{T} points assigned to a preallocated array
"""
function auto_corr!(r::AbstractVector{T}, histo::AbstractVector{Int}, rmax::Int, nbins::Int; blocksize::Int=64, metric::Metric=Euclidean()) where {T<:SVector}
    N = length(r) #this seems to be triggering an eval
    if N <= blocksize
        return _auto_corr!(r,histo, rmax, nbins, metric)
    end
    chunks = cld(N, blocksize)
    tiles = SplitAxis(1:N, chunks)
    ntile = Tuple{UnitRange{Int64}, UnitRange{Int64}}[]
    for i in 1:length(tiles)
        for j in i+1:length(tiles)
            push!(ntile, (tiles[i], tiles[j]))
        end
    end
    histos = [zeros(Int64, length(histo)) for i in 1:Threads.nthreads()]  
    @sync Threads.@threads for i in tiles
        _auto_corr!(view(r,i), view(histos, Threads.threadid())[1], rmax, nbins, metric) #
    end
    @sync Threads.@threads for (i,j) in ntile
        _cross_corr!(view(r,i), view(r,j), view(histos, Threads.threadid())[1], rmax, nbins, metric=metric)
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

export auto_corr, cross_corr, unitary_corr, auto_corr!, cross_corr!, unitary_corr!


end

# TODO: Why is rmax passed as an Int? It should probably be a ::Real