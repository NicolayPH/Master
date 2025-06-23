#=
Want to create samples for use in neural network
The samples are four dimensional and contains P, T, x1 and x2
x3 is a function of x1 and x2
=#

using Surrogates, Plots, QuasiMonteCarlo
#Reject the points here

function constrained_samples(number_samples::Int, lb_p::Float64, lb_T::Float64, ub_p::Float64, ub_T::Float64)
    valid_samples = []
    while length(valid_samples) < number_samples
        samples = QuasiMonteCarlo.sample(number_samples*2, [lb_p, lb_T, 0.0, 0.0], [ub_p, ub_T, 1.0, 1.0], LatinHypercubeSample())
        p = samples[1,:]
        t = samples[2,:]
        x1 = samples[3,:]
        x2 = samples[4,:]
        #filter valid points
        for i in 1:length(x1)
            if x1[i] .+ x2[i] <= 1
                push!(valid_samples, [p[i], t[i], x1[i], x2[i]])
                if length(valid_samples) == number_samples
                    break
                end
            end
        end
    end
    return valid_samples
end


function PT_plot(v)
    P = v[1]
    T = v[2]

    scatter(T, P, xlabel = "Temperature [K]", ylabel = "Pressure [Pa]", label = "Samples");
end

function Tx1_plot(v)

    T = v[2]
    x1 = v[3]
    #x2 = v[4]
    #x1 = x1 ./ (x1 .+ x2);   #In order to not get x1 + x2 > 0
    scatter(T, x1, xlabel = "Temperature [K]", ylabel = "x1", label = "Samples")
end

function Px1_plot(v)
    P = v[1]
    x1 = v[3]
    x2 = v[4]
    #x1 = x1 ./ (x1 .+ x2)
    #x2 = x2 ./ (x1 .+ x2)

    scatter(P, x1, xlabel = "Pressure [Pa]", ylabel = "x1", label = "Samples")
end

function x1x2_plot(v)
    x1 = v[3]
    x2 = v[4]

    scatter(x1, x2, xlabel = "x1", ylabel = "x2", label = "Samples")
end

#PT_plot(samples)
#Tx1_plot(samples)
#Px1_plot(samples)
