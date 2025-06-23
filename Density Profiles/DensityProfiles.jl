using Clapeyron, cDFT
using Plots

model = PCSAFT(["hydrogen", "carbon dioxide", "nitrogen"])

surface = Steele(["graphite"])
CO2_model = PCSAFT(["carbon dioxide"])
L = length_scale(CO2_model)
H = 50 * 10^(-10) #https://doi.org/10.3390/e27020184
bounds = [0.82L, H - 0.82L]

device = DFTOptions(cDFT.CPU(), cDFT.AndersonFixPoint(picard_damping=0.0001, damping=0.001, drop_tol=1e1))

function calculate_mixture(P, T, n, model, device = DFTOptions(cDFT.CPU(), cDFT.AndersonFixPoint(picard_damping=0.0001, damping=0.001, drop_tol=1e1)))
    v = volume(model, P, T, n)
    ρ = n./v
    structure = ExternalField1DCart((P, T), ρ, bounds, (201,), surface, H)

    system = DFTSystem(model, structure, device);
    ρ_ = cDFT.initialize_profiles(system);
    converge!(system, ρ_);

    return ρ_, system
end

x = range(1, 201, 201) |> collect;
function plot_mixture(ρ_, x)
    fig = plot()
    plot!(x, ρ_, xlabel = "Position", ylabel = "Density", label = ["Hydrogen" "Carbon Dioxide" "Nitrogen"])
    fig
end

#plot_mixture(ρ_, x)