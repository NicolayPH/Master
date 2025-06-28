using Flux
using Plots

z = range(-5, 5, length=1000) |> collect
a₁ = σ.(z)
a₂ = tanh.(z)

default(fontfamily = "Computer Modern", guidefontsize = 12, tickfontsize = 12, legendfontsize = 10, framestyle = :default, margin = 6Plots.mm)

p₁ = plot(z, a₁, xlabel = "z", ylabel = "a", label = "σ", lw = 2, legend =:topleft, size = (600, 400))
p₂ = plot(z, a₂, xlabel = "z", label = "tanh")

plt = plot(p₁, p₂, layout=(1,2), size=(1000,350), dpi=300)

savefig(p, "SigmaFunction.pdf")