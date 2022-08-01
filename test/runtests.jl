using DistanceHistograms, StaticArrays, Distances, Test
import DistanceHistograms.as_ints, DistanceHistograms.svectorscopy

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
c = svectorscopy(Float64.(as_ints(vec(a))), Val{2}())
d = copy(b)

#pack the metric type with the maximally distant points in the set
euc = disteval(SVector{2}([1.0, 1.0]), SVector{2}([100.0, 100.0]), Euclidean())
peuc = disteval(SVector{2}([1.0, 1.0]), SVector{2}([100.0, 100.0]), Cityblock())
cb = disteval(SVector{2}([1.0, 1.0]), SVector{2}([100.0, 100.0]), Chebyshev())
cheby = disteval(SVector{2}([1.0, 1.0]), SVector{2}([50.0, 50.0]), PeriodicEuclidean([100,100]))


metrics = (euc, peuc, cb, cheby)
maxdists = [distance(x) for x=metrics]
resb = [auto_corr(b, ceil(Int64, maxdist)+1, 100, blocksize=50, metric=met.f) for (met , maxdist) in zip(metrics,maxdists)]
resc = [auto_corr(c, ceil(Int64, maxdist)+1, 100, blocksize=50, metric=met.f) for (met , maxdist) in zip(metrics,maxdists)]
resd = [cross_corr(b, b, ceil(Int64, maxdist)+1, 100, blocksize=50, metric=met.f) for (met , maxdist) in zip(metrics,maxdists)]
rese = [cross_corr(c, c, ceil(Int64, maxdist)+1, 100, blocksize=50, metric=met.f) for (met , maxdist) in zip(metrics,maxdists)]
mins = [77616, 97410, 39402, 40000] # second bins of each histogram

@testset "Autocorrelation tests" begin
    @testset "Autocorrelation reinterpret distance totals" begin
        dt = tri(length(a))
        @test sum(resb[1])==dt
        @test sum(resb[2])==dt
        @test sum(resb[3])==dt
        @test sum(resb[4])==dt
    end

    @testset "Autocorrelation reinterpret distance sample" begin
        @test resb[1][2] == mins[1]
        @test resb[2][2] == mins[2]
        @test resb[3][2] == mins[3]
        @test resb[4][2] == mins[4]
    end

    @testset "Autocorrelation SVector distance totals" begin
        dt = tri(length(a))
        @test sum(resc[1])==dt
        @test sum(resc[2])==dt
        @test sum(resc[3])==dt
        @test sum(resc[4])==dt
    end

    @testset "Autocorrelation SVector distance sample" begin
        @test resc[1][2] == mins[1]
        @test resc[2][2] == mins[2]
        @test resc[3][2] == mins[3]
        @test resc[4][2] == mins[4]
    end
end

@testset "Cross correlation tests" begin

    @testset "Cross correlation reinterpret distance totals" begin
        dt = length(a)^2
        @test sum(resd[1])==dt
        @test sum(resd[2])==dt
        @test sum(resd[3])==dt
        @test sum(resd[4])==dt
    end

    @testset "Cross correlation reinterpret distance sample" begin
        @test resd[1][2] == mins[1]*2
        @test resd[2][2] == mins[2]*2
        @test resd[3][2] == mins[3]*2
        @test resd[4][2] == mins[4]*2
    end

    @testset "Cross correlation SVector distance totals" begin
        dt = length(a)^2#tri(length(a))
        @test sum(rese[1])==dt
        @test sum(rese[2])==dt
        @test sum(rese[3])==dt
        @test sum(rese[4])==dt
    end

    @testset "Cross correlation SVector distance sample" begin
        @test rese[1][2] == mins[1]*2
        @test rese[2][2] == mins[2]*2
        @test rese[3][2] == mins[3]*2
        @test rese[4][2] == mins[4]*2
    end

end