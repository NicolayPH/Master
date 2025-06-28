using DelimitedFiles  
using LinearAlgebra
using Statistics
using Plots
default(fontfamily="Computer Modern", guidefontsize=12, tickfontsize=12, legendfontsize=10, framestyle=:default, margin=5Plots.mm)
using LaTeXStrings
using ReservoirComputing
using OrdinaryDiffEq
using DataInterpolations
using MLJLinearModels
import Random
using Surrogates
using Random
using Clapeyron, cDFT
using BenchmarkTools
#using SparseArrays


# Reading and preprocessing the data
function read_data(file_path::String)
    data = readdlm(file_path, ',')
    return data
end

X = read_data("Samples.csv")
Y = read_data("DensityProfilesCarbonDioxide.csv")[:, 1:101]


function find_reference_solution(X, Y, f::Function)
    # Assuming the reference solution is the first column of X
    f_x = f(X[:, 1])
    e = abs.(X .- f_x)
    # Find the index of the minimum error
    min_index = argmin(e)[1]
    return min_index, X[min_index, :], Y[min_index, :]
end 

idx, X__, Y__ =  find_reference_solution(X, Y, mean)

    


function read_matrix(filename)
    matrices = []
    current_matrix = []
    open(filename, "r") do f
        for line in eachline(f)
            line_stripped = strip(line)
            if isempty(line_stripped)
                push!(matrices, current_matrix)
                current_matrix = []
            else
                row = parse.(Float64, split(line_stripped, ','))
                push!(current_matrix, [row])
            end
        end
        push!(matrices, current_matrix)  # Add last matrix
    end
    return matrices
end

_X = X[1:end .!= idx, :]
_Y = Y[1:end .!= idx, :]
idxs = Int.(range(1, 499, 499))
shuffled_indexes = shuffle(idxs)

_X = _X[shuffled_indexes, :]
_Y = _Y[shuffled_indexes, :]

# Split the data into training and testing sets
function split_data(X, Y, train_ratio)
    n = size(X, 1)
    train_size = Int(round(n * train_ratio))
    X_train = X[1:train_size, :]
    Y_train = Y[1:train_size, :]
    X_test = X[train_size+1:end, :]
    Y_test = Y[train_size+1:end, :]
    return X_train, Y_train, X_test, Y_test
end

function fit_data_normalization(dataset; dims = 1)

    if !isnothing(dims)
        # Fit the normalization parameters
        μ_X = mean(dataset, dims = dims)
        σ_X = std(dataset, dims = dims)
    else
        # Fit the normalization parameters
        μ_X = mean(dataset)
        σ_X = std(dataset)
    end

    return μ_X, σ_X
end



function apply_normalization(X, μ_X, σ_X)
    # Normalize the data
    X_normalized = (X .- μ_X) ./ σ_X
    return X_normalized
end

X_train, Y_train, X_test, Y_test = split_data(X, Y, 0.8);

X_train_scaled = apply_normalization(X_train, fit_data_normalization(X_train, dims = 1)...)
μ_Y, σ_Y = fit_data_normalization(Y_train, dims = nothing) 
Y_train_scaled = apply_normalization(Y_train, μ_Y, σ_Y)
#plot(Y_train_scaled[10, :])

X_test_scaled = apply_normalization(X_test, fit_data_normalization(X_train, dims = 1)...)
Y_test_scaled = apply_normalization(Y_test, μ_Y, σ_Y)

t = range(1.0, 101, length = 101) |> collect
Y__scaled = apply_normalization(Y__, μ_Y, σ_Y)
#sol_ref = QuadraticInterpolation(Y__scaled, t)
sol_ref = BSplineInterpolation(Y__scaled, t,  3, :ArcLen, :Average)
plot(sol_ref, xlabel = "z", ylabel = "Scaled Density Values")
savefig("B-spline interpolation.pdf")

#t = range(1.0, 101.0, length = 101) |> collect 


plot(sol_ref)

#CTESN
reservoirSize = 400

W = rand_sparse(reservoirSize, reservoirSize, sparsity = 0.1)
Win = weighted_init(reservoirSize, 1)

function CTESN!(du, u, p, t)
	du .= p[1].*tanh.(p[2]*u .+ p[3]*p[4](t)) .- (1.0 - p[1]).*u # p[3](t; idxs=49:49)
	return nothing
end

rng = Random.seed!(1234)
u0 = randn(rng, Float64, reservoirSize)
esnprob = ODEProblem(CTESN!, u0, (1.0, 101.0), (0.8, W, Win, sol_ref))

r = solve(esnprob, AutoTsit5(Vern7()); saveat = t, abstol = 1e-8, reltol = 1e-8) #Reservoir dynamics
    
plot(r)
# Compute Wout for reference solution

Wout = [zeros(Float64, 1, reservoirSize + 1)]

function fitData!(Wout, xData, rData; beta = 0.001, iterative = iterative)
	fitModel = RidgeRegression(beta) 
	#=
	Note:
		rData = hcat(r(sol.t)...)
		xData = hcat(sol(sol.t)...)
	But we avoid re-interpolating them every time by requiring the r solution use `saveat=sol.t`
	=#

    nrows = ifelse(length(size(xData)) == 1, 1, size(xData, 1))
    println("nrows = ", nrows)

	for n ∈ 1:nrows
        W_fit = fit(fitModel, transpose(rData[:,:]), xData[n, :]; solver=Analytical(iterative=iterative))
        println("W_fit = ", size(W_fit))
	    #tmp = transpose(W_fit)
	    Wout[n][1, :] .= W_fit
    end
	return nothing
end


# Testing the fitting
#=
fitData!(Wout, Y_train_scaled[200:200, :], r; beta = 1e-3, iterative = true)
r_1 = vcat(r[:, :], ones(1, length(t))) 
Y__pred = Wout[1]*r_1 
plot(Y__pred[:], label = "Predicted")
scatter!(Y_train_scaled[200, :], label = "True")
scatter(Y__pred[:], Y__scaled, label = "Predicted vs True")
=#

# Compute Wout for all training data

Wouts = [zeros(Float64, 1, reservoirSize + 1) for _ in 1:size(X_train_scaled, 1)]

fitData!(Wouts, Y_train_scaled, r; beta = 1e-3, iterative = true)

X_train_scaled_  = [X_train_scaled[i, :] for i in 1:size(X_train_scaled, 1)]
X_test_scaled_ = [X_test_scaled[i, :] for i in 1:size(X_test_scaled, 1)]

u_lower = minimum(X_train_scaled, dims = 1)
u_upper = maximum(X_train_scaled, dims = 1)

test_profile = zeros(size(Y_test_scaled))

WoutInterpolant = RadialBasis(X_train_scaled_, Wouts, u_lower, u_upper)
Wouts_test1 = WoutInterpolant(X_test_scaled_[1])
test_profile[1,:] = transpose(r_1)*Wouts_test1
p = scatter(test_profile[1,:], Y_test_scaled[1, :], label = false)



lowest_idx = 1
largest_idx = 1
#MSE error
lowest_error = sum((test_profile[1,:] .- Y_test_scaled[1,:]).^2)
largest_error = sum((test_profile[1,:] .- Y_test_scaled[1,:]).^2)

#MAPE_error
lowest_error = sum((abs.((test_profile[1,:] .- Y_test_scaled[1,:])./Y_test_scaled[1,:])))/length(test_profile[1,:])
largest_error = sum((abs.((test_profile[1,:] .- Y_test_scaled[1,:])./Y_test_scaled[1,:])))/length(test_profile[1,:])


for i in 2:length(X_test_scaled_)
    Wouts_test = WoutInterpolant(X_test_scaled_[i])
    test_profile[i,:] = transpose(r_1)*Wouts_test
    predicted = test_profile[i,:]
    actual = Y_test_scaled[i,:]


    #Want to find the density profiles with the smallest and largest errors
    MAPE_error = sum((abs.((predicted .- actual)./actual)))/length(predicted)
    #mse = sum((predicted-actual).^2)/length(predicted)
    println(MAPE_error)
    if MAPE_error < lowest_error
        lowest_error = MAPE_error
        lowest_idx = i
    elseif MAPE_error > largest_error
        largest_error = MAPE_error
        largest_idx = i
    end
    scatter!(p, test_profile[i,:], Y_test_scaled[i, :], label = false)
end


xmin, xmax = xlims(p)
plot!(p, [xmin, xmax], [xmin, xmax], label = false, line =:dash, color = :black)
plot!(p, title = "Carbon Dioxide CTESN", xlabel = "Scaled True Value", ylabel = "Scaled Predictions")
display(p)
savefig(p, "CarbonDioxide_CTESN101_updated.pdf")

lowest_idx
largest_idx

largest_error*100
lowest_error*100


# Plot 1
p1 = plot(
    Y_test_scaled[lowest_idx,:],
    label = "True",
    xlabel = "z",
    ylabel = "Normalized Density",
    title = "Lowest Error Carbon Dioxide",
    lw = 2,
    legend= (0.7, 0.9)
)
scatter!(test_profile[lowest_idx,:], label = "Predicted")

# Plot 2 
p2 = plot(
    Y_test_scaled[largest_idx,:],
    label = "True",
    xlabel = "z",
    ylabel = "",  # No y-axis label to avoid repetition
    title = "Largest Error Carbon Dioxide",
    lw = 2,
    legend= (0.7, 0.9),
)
scatter!(test_profile[largest_idx,:], label = "Predicted")

# Combine
plt = plot(p1, p2, layout=(1,2), size=(1000,350), dpi=300)

savefig(plt, "CarbonDioxide_density_errors.pdf") 





#Want to test the computation time for simple cDFT calculations and CTESN
function calculate_mixture_cDFT(P, T, n, device = DFTOptions(cDFT.CPU(), cDFT.AndersonFixPoint(picard_damping=0.0001, damping=0.001, drop_tol=1e1)))
    #Definining constants for the calculations
    model = PCSAFT(["hydrogen", "carbon dioxide", "nitrogen"])
    surface = Steele(["graphite"])
    CO2_model = PCSAFT(["carbon dioxide"])
    L = length_scale(CO2_model)
    H = 50 * 10^(-10) #https://doi.org/10.3390/e27020184
    bounds = [0.82L, H - 0.82L]

    v = volume(model, P, T, n)
    ρ = n./v
    structure = ExternalField1DCart((P, T), ρ, bounds, (101,), surface, H)

    system = DFTSystem(model, structure, device);
    ρ_ = cDFT.initialize_profiles(system);
    converge!(system, ρ_);

    return ρ_, system
end

testing_params = X_test[1,:]
@btime calculate_mixture_cDFT(testing_params[1], testing_params[2], [testing_params[3], testing_params[4], 1 .- testing_params[3] .- testing_params[4]])
@btime transpose(r_1)*WoutInterpolant(X_test_scaled_[1])

plot(test_profile[1,:], label = "Predicted")
#scatter(test_profile[1,:], Y_test_scaled[1, :])
scatter!(Y_test_scaled[1, :], label = "True")


