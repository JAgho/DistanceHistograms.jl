# DistanceHistograms.jl

DistanceHistograms.jl is a lightweight package for optimised computation of pairwise and cross-correlated distance histograms. These summarise distance matrix distributions with constant memory and greater efficiency than compute-then-bin approaches. Inputs are collections of coordinates, given as `Vector{SVector}`; outputs are `Vector{Int64}`. All Distances.jl `SemiMetric`s are supported, with some specialised versions available. Binning is linear and 0->maximum style.  This packaged is optimised for medium scale 2-point correlation function computation, and does not employ spatial tree techniques, though the kernels are suitable for this.

Compute an autocorrelation function with `auto_corr`, or a cross correlation function with `cross_corr`. Select a suitable maximal value by setting `rmax`, and the number of bins with `nbins`. `metric` may be set to any `SemiMetric` to modify the distance computation used. `blocksize` may be used to set the batch size for autocorrelation computation.

We can compute a distance histogram over any set of points, 
```julia-repl
# generate a set of points
julia> a = collect(CartesianIndices((100,100)));
# convert to SVectors
julia> b = reinterpret(SVector{length(axes(a)),Float64}, Float64.(vec(DistanceHistograms.as_ints(a))));
#computing a correlation function with 10 bins
julia> auto_corr(b,zeros(Int64, 10), 145, 10, metric=Euclidean())
10-element Vector{Int64}:
 2790126
 6994502
 9263358
 9780918
 8944602
 6970206
 3913838
 1113342
  212062
   12046

julia> cross_corr(b,b,zeros(Int64, 10), 145, 10, metric=Euclidean())
10-element Vector{Int64}:
  5590252
 13989004
 18526716
 19561836
 17889204
 13940412
  7827676
  2226684
   424124
    24092

```
