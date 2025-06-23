using Surrogates, Plots, Surrogates

function constrained_lhs(n::Int)
    valid_samples = []
    while length(valid_samples) < n
        s = sample(n*2, [0.0, 0.0], [1.0, 1.0], LatinHypercubeSample())  # Generate more points than needed
        x1 = [x[1] for x in s]
        x2 = [x[2] for x in s]
        # Filter valid points
        for i in 1:length(x1)
            if x1[i] .+ x2[i] <= 1
                push!(valid_samples, [x1[i], x2[i]])
                if length(valid_samples) == n
                    break
                end
            end
        end
    end
    return valid_samples
end

function x1x2(n::Int)
    samples = constrained_lhs(n)

    x1 = [x[1] for x in samples];
    x2 = [x[2] for x in samples];
    return x1, x2
end

function x_plot(x1::Vector{Float64}, x2::Vector{Float64})
    #plotly()
    x3 = 1 .- x1 .- x2;

    scatter(x1, x2, xlabel = "x1", ylabel = "x2", title = "Samples")
    #scatter3d(x1, x2, x3, xlabel = "x1", ylabel = "x2", zlabel = "x3", title = "Samples")
end

