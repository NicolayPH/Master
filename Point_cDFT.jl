include("DensityProfiles.jl")
include("WriteToCSV.jl")
using DelimitedFiles

function Point_cDFT(vec, model = PCSAFT(["hydrogen", "carbon dioxide", "nitrogen"]))
    
    number_iterations = length(vec[1])
    P = vec[1]
    T = vec[2]
    x1 = vec[3]
    x2 = vec[4]
    x3 = 1 .- x1 .- x2

    valid_iters = []
    #invalid_iters = []
    for i in 1:number_iterations
        ρ__, _ =  calculate_mixture(P[i], T[i], [x1[i], x2[i], x3[i]], model)
        push!(valid_iters, ρ__)
        println(i)
    end
        return valid_iters#, invalid_iters
end


array
mixture_model = PCSAFT(["hydrogen", "carbon dioxide", "nitrogen"])
densities = Point_cDFT(array)

densities_hydrogen = [d[:,1] for d in densities]
densities_carbonDioxide = [d[:,2] for d in densities]
densities_nitrogen = [d[:,3] for d in densities]

filename_hydrogen = "DensityProfilesHydrogen.csv"
filename_carbonDioxide = "DensityProfilesCarbonDioxide.csv"
filename_nitrogen = "DensityProfilesNitrogen.csv"
writedlm(filename_hydrogen, densities_hydrogen, ',')
writedlm(filename_carbonDioxide, densities_carbonDioxide, ',')
writedlm(filename_nitrogen, densities_nitrogen, ',')


