using Plots
using StatsPlots
using Distributions

default(fontfamily="Computer Modern", guidefontsize=12, tickfontsize=12, legendfontsize=10, framestyle=:default, margin=6Plots.mm, dpi=300)

actual_values = randn(100)
predicted_values = actual_values .+ 0.3 * randn(100)
errors = predicted_values .- actual_values
sample_values = randn(100)

p = histogram(errors, bins=20, xlabel="Error", ylabel="Frequency", label=false, title="Prediction Error Histogram", size=(600,400))
savefig(p, "Error_histogram.pdf")

p₂ = scatter(actual_values, predicted_values, xlabel="Observed values", ylabel="Predicted values", label=false, title="Parity plot", size=(600,400))
xₗ = xlims(p₂)
plot!(p₂, [xₗ[1], xₗ[2]], [xₗ[1], xₗ[2]], line=:dash, label=false)
savefig(p₂, "Parity_plot.pdf")

qq = qqplot(sample_values, Normal(0,1), xlabel="Theoretical Quantiles", ylabel="Sample Quantiles", title="Q-Q Plot", size=(600,400))
savefig(qq, "Q-Q_plot.pdf")

