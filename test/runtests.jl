using DistanceHistograms


using StaticArrays, Distances, TiledIteration

@inline function cross_dhist(r1, r2, histo, rmax::Int64, nbins::Int64, dist)
    rb = nbins/rmax
    metric = dist()
    @inbounds for i in 1:length(r1)
        @inbounds for j in 1:length(r2)
            d=evaluate(metric, r1[i], r2[j])#sqrt((r1[i,1]-r2[j,1])^2+(r1[i,2]-r2[j,2])^2)
            rIndex=floor(Int, d*rb) + 1
            (rIndex < nbins+1) && (histo[rIndex] += 1)
        end
    end
    return histo
end

function cross_dhist2(r1::AbstractArray{Float64,2}, r2::AbstractArray{Float64,2}, histo, rmax::Int64, nbins::Int64)
    rb = nbins/rmax
    @inbounds for i in 1:size(r1)[2]
        for j in 1:size(r2)[2]
            d=sqrt((r1[1,i]-r2[1,j])^2+(r1[2,i]-r2[2,j])^2)
            rIndex=floor(Int, d*rb) + 1
            (rIndex < nbins+1) && (histo[rIndex] += 1)
        end
    end
    return histo
end


@inline function self_dhist(r, histo, rmax, nbins, dist)
    rb = (nbins)/rmax 
    metric= dist()
    @inbounds for i = 1:length(r)-1
        for j = i+1:length(r)#i+1:size(r)[1]
            d=evaluate(metric, r[i], r[j])
            rIndex=floor(Int, d*rb) +1
            (rIndex < nbins+1) && (histo[rIndex] += 1)
        end
    end
    return histo
end

function self_dhist2(r::AbstractArray{Float64,2}, histo, rmax, nbins)
    rb = (nbins)/rmax 
    @inbounds for i = 1:size(r)[2]-1
        for j = i+1:size(r)[2]
            d=sqrt((r[1,i]-r[1,j])^2+(r[2,i]-r[2,j])^2)
            rIndex=floor(Int, d*rb) +1
            (rIndex < nbins+1) && (histo[rIndex] += 1)
        end
    end
    return histo
end


function tile_dist(points, histo::Array{Int64,1}, rmax::Int64, nbins::Int64; blocksize=50, metric=Euclidean)
    N = length(points)
    if N <= blocksize
        return self_dhist(b,histo, rmax, nbins, metric)
    end
    chunks = length(points) ÷ blocksize
    tiles = collect(SplitAxis(1:N, chunks))
    ntile = Tuple{UnitRange{Int64}, UnitRange{Int64}}[]
    for i in 1:length(tiles)
        for j in i:length(tiles)
            push!(ntile, (tiles[i], tiles[j]))
        end
    end
    histos = [zeros(Int64, length(histo)) for i in 1:Threads.nthreads()]  
    @sync Threads.@threads for i in tiles
        #println(i)
        self_dhist(view(points,i), view(histos, Threads.threadid())[1], rmax, nbins, metric) #
    end
    #println("\npause\n")
    @sync Threads.@threads for (i,j) in ntile
        #println(i, "  ", j)
        cross_dhist(view(points,i), view(points,j), view(histos, Threads.threadid())[1], rmax, nbins, metric)
    end 
    return sum(histos)
end


function tile_dist(points, histo::Array{Int64,1}, rmax::Int64, nbins::Int64)
    bsize::Int64 = 100
    N = size(points)[1]
    #histo .= 0
    if N <= bsize
        return self_dhist2(points, histo, rmax, nbins)
    else
        histos = [zeros(Int64, length(histo)) for i in 1:Threads.nthreads()]
        #dump(histos[1])
        Threads.@threads for i in 0:(div(N, bsize)-1)
                            self_dhist2(view(points, (i*bsize)+1:((i+1)*bsize), :),
                                 view(histos, Threads.threadid())[1],
                                 rmax,
                                 nbins)
            #print("\nirange = ", (i*bsize)+1, ":", ((i+1)*bsize))
            #make thread with self_routine(myblocks)
        end
        @sync for il in 1:bsize:N+1
            for jl in il+bsize:bsize:N-bsize+1
                #println("il: ", il, "\t\tir: ", il+bsize-1)
                #println("jl: ", jl, "\t\tjr: ", jl+bsize-1)
                Threads.@spawn cross_dhist2(view(points, il:il+bsize-1, :),
                        view(points, jl:jl+bsize-1, :),
                        view(histos, Threads.threadid())[1],
                        rmax,
                        nbins)
            end
        end
        epoint = (N ÷ bsize)*bsize
        #println("epoint = ", epoint, ", residuals are: ", epoint+1:N, " and  ", 1:epoint)
    self_dhist2(view(points, epoint+1:N, :),
         view(histos, 1)[1],
         rmax,
         nbins)
     cross_dhist2(view(points, epoint+1:N, :),
             view(points, 1:epoint, :),
             view(histos, Threads.threadid())[1],
             rmax,
             nbins)
        for i in 1:Threads.nthreads() histo .+= view(histos, i)[1] end
    end
    return histo
end

a = collect(CartesianIndices((100,100)));
b = reinterpret(SVector{length(axes(a)),Float64}, Float64.(vec(as_ints(a))));
c = reshape(Float64.(vec(as_ints(a))), 2, :)

# self_dhist(b,zeros(Int64, 100), 145, 100, Euclidean)
# self_dhist2(c,zeros(Int64, 100), 145, 100)

# cross_dhist(b,b,zeros(Int, 100), 145, 100, Euclidean)
# cross_dhist2(c,c,zeros(Int, 100), 145, 100)

auto_corr(b,zeros(Int64, 100), 145, 100)
#temp2 = tile_dist(c,zeros(Int64, 100), 145, 100)
#temp3 = self_dhist2(c,zeros(Int64, 100), 145, 100)


tiles = collect(SplitAxis(1:length(b), 8))
ntile = Tuple{UnitRange{Int64}, UnitRange{Int64}}[]
for i in 1:length(tiles)
    for j in i:length(tiles)
        push!(ntile, (tiles[i], tiles[j]))
    end
end

inner = (2:8, 2:8)
outer = (1:10, 1:10)
i3 = (2:8, 2:8, 2:8)
o3 = (1:10, 1:10, 1:10)
ei = EdgeIterator(o3, i3)

import Base.Cartesian.@nloops, Base.Cartesian.@ntuple




dims = length(axes(ei.inner))
ends(x) = (first(x), last(x))
s = size(zeros(100,90,80, 70,60))
dims = length(s)
v = 2^(dims-1)
vars = [zeros(v, i) for i in s]
vars[2] #hurrah this generates the edges of a n-rectangle... now to populate it





edges = [zeros(i) for i in 1:2^(dims-1)*2]
for (i,e) in enumerate(EdgeIterator(outer, inner))
    @show e
end

