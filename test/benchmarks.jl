using DistanceHistograms, StaticArrays, Distances, BenchmarkTools
import DistanceHistograms.as_ints, DistanceHistograms.svectorscopy, StatsBase.fit

# test struct to pack metrics with a coordinate pair
struct disteval{T<:SVector, M<:SemiMetric}
    a::T
    b::T
    f::M
end
distance(x::disteval) =  Distances.evaluate(x.f, x.a, x.b)
tri(n::Int64) = n*(n-1)÷2

# create test array of 100x100 isotropically distributed points in svector format
a = collect(CartesianIndices((100,100)));
b = reinterpret(SVector{length(axes(a)),Float64}, Float64.(vec(as_ints(a))));
#c = svectorscopy(Float64.(as_ints(vec(a))), Val{2}())
c = copy(b)

#pack the metric type with the maximally distant points in the set
euc = disteval(SVector{2}([1.0, 1.0]), SVector{2}([100.0, 100.0]), Euclidean())
cb = disteval(SVector{2}([1.0, 1.0]), SVector{2}([100.0, 100.0]), Cityblock())
cheby = disteval(SVector{2}([1.0, 1.0]), SVector{2}([100.0, 100.0]), Chebyshev())
peuc = disteval(SVector{2}([1.0, 1.0]), SVector{2}([50.0, 50.0]), PeriodicEuclidean([100,100]))


metrics = (euc, cb, cheby, peuc)
maxdists = [distance(x) for x=metrics]
#res = [auto_corr(b,zeros(Int64, 100), ceil(Int64, maxdist)+1, 100, blocksize=50, metric=met.f) for (met , maxdist) in zip(metrics,maxdists)]
#mins = [77616, 97410, 39402, 40000] # second bins of each histogram
e = Euclidean()
cb = Cityblock()
ch = Chebyshev()
pe = PeriodicEuclidean([100.0,100.0])

### reinterpreted SVector test

#Matlab equivalent computation using pdist and histcounts was ~138ms
@benchmark auto_corr!(b,zeros(Int64, 100), ceil(Int64, maxdists[1])+1, 100, blocksize=1000, metric=e)
#Euclidean = 22.7ms
@benchmark auto_corr!(b,zeros(Int64, 100), ceil(Int64, maxdists[2])+1, 100, blocksize=1000, metric=cb)
#Cityblock 15.9ms
@benchmark auto_corr!(b,zeros(Int64, 100), ceil(Int64, maxdists[3])+1, 100, blocksize=1000, metric=ch)
#Chebyshev 21.5ms
@benchmark auto_corr!(b,zeros(Int64, 100), ceil(Int64, maxdists[4])+1, 100, blocksize=200, metric=pe)
#Periodic Euclidean 81.5ms -looks like this is just a tough computation


### pure SVector test

@benchmark auto_corr!(c,zeros(Int64, 100), ceil(Int64, maxdists[1])+1, 100, blocksize=1000, metric=e)
#Euclidean = 21.2ms
@benchmark auto_corr!(c,zeros(Int64, 100), ceil(Int64, maxdists[2])+1, 100, blocksize=1000, metric=cb)
#Cityblock 14.9ms
@benchmark auto_corr!(c,zeros(Int64, 100), ceil(Int64, maxdists[3])+1, 100, blocksize=1000, metric=ch)
#Chebyshev 20.3ms
@benchmark auto_corr!(c,zeros(Int64, 100), ceil(Int64, maxdists[4])+1, 100, blocksize=200, metric=pe)
#Periodic Euclidean 77.4ms


# this is much faster than computing the distance matrix and binning
# @btime R = pairwise(Euclidean(), c)
# @btime R = pairwise(Euclidean(), c, c)
# #291ms
# @btime fit(Histogram, vec(R), 0.0:1.45:145.0)
# #2692ms


