using Surrogates
using Flux
using SurrogatesFlux
using DelimitedFiles
using StatsBase
using Random 
using Plots
using CSV

filename = "Samples.csv"
samples = readdlm(filename, ',')

P = samples[:, 1]
T = samples[:, 2]
min_P = minimum(P)
max_P = maximum(P)
min_T = minimum(T)
max_T = maximum(T)

#Want to normalize pressure and Temperature
P = (P .- min_P)/(max_P - min_P)
T = (T .- min_T)/(max_T- min_T)
x1 = samples[:, 3]
x2 = samples[:, 4]

#Want to create the training and test data
densities_hydrogen = readdlm("DensityProfilesHydrogen.csv", ',')
densities_carbonDioxide = readdlm("DensityProfilesCarbonDioxide.csv", ',')
densities_nitrogen = readdlm("DensityProfilesNitrogen.csv", ',')
densities_component = Dict()
densities_component["hydrogen"] = densities_hydrogen
densities_component["carbon dioxide"] = densities_carbonDioxide
densities_component["nitrogen"] = densities_nitrogen


P = reshape(P, :, 1)
T = reshape(T, :, 1)
x1 = reshape(x1, :, 1)
x2 = reshape(x2, :, 1)

xys = hcat(P, T, x1, x2)'
xys = Matrix(xys)


function remove_nan_columns(mat, removed_index = 0)
  if removed_index != 0
    mask = trues(size(mat, 2))
    mask[removed_index] .= false  #set the indices we want to remove to false
    removed_index = 0
    return mat[:, mask], removed_index
  else
    mask = vec(.!any(isnan, mat; dims=1))  # Convert row vector to a 1D array
    removed_index = findall(.!mask) #Indices of the removed columns
  return mat[:, mask], removed_index   # Use the mask to filter columns
  end
end

zs = densities_component["nitrogen"]
zs = zs'
zs = Matrix(zs)
zs, removed_index = remove_nan_columns(zs)
xys, removed_index = remove_nan_columns(xys, removed_index)


divide_60 = Int(round(0.60 * length(xys[1, :])))

shuffled_index = shuffle(1:length(xys[1, :]))

xys = xys[:, shuffled_index]

zs = zs[:, shuffled_index]

input = xys[:, 1:divide_60]
validation_input = xys[:, divide_60+1:end]
label = zs[:, 1:divide_60]
validation_label = zs[:, divide_60+1:end]

filename = "PreparedTrainingNitrogen.csv"

writedlm(filename, input, ',')

open(filename, "a") do io 
  writedlm(io, " ")
  writedlm(io, validation_input, ',')
  writedlm(io, " ")
  writedlm(io, label, ',')
  writedlm(io, " ")
  writedlm(io, validation_label, ',')
end


