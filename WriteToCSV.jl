include("Samples.jl")
include("x1x2Samples.jl")
using DelimitedFiles

n = 500
lb_p = 1.0
lb_t = 293.15
ub_p = 100e5
ub_t = 473.15

samples = constrained_samples(n, lb_p, lb_t, ub_p, ub_t)

P = [samples[i][1] for i in 1:length(samples)]
T = [samples[i][2] for i in 1:length(samples)]
x1 = [samples[i][3] for i in 1:length(samples)]
x2 = [samples[i][4] for i in 1:length(samples)]

array = [P, T, x1, x2];

#Tx1_plot(array)
#PT_plot(array)
#Px1_plot(array)
#x1x2_plot(array)


filename = "Samples.csv"
writedlm(filename, samples, ',')