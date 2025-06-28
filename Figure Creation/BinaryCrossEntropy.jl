using Plots
using LaTeXStrings

default(fontfamily = "Computer Modern", guidefontsize = 12, tickfontsize = 12, legendfontsize = 10, framestyle = :default, margin = 6Plots.mm)

# Define x avoiding 0 and 1 (to prevent -Inf)
x = range(0.001, 0.999, length=1000) |> collect

# Compute losses
loss1 = -log.(x)
loss2 = -log.(1 .- x)

# Plot
p = plot(x, loss1, label = L"-\log(f)", xlabel = L"f", ylabel = L"J", lw = 2, legend = (0.6, 0.9), size = (600, 400), dpi = 300)
plot!(p, x, loss2, label = L"-\log(1 - f)")

# Save final styled figure
savefig(p, "BinaryCrossEntropy.pdf")
