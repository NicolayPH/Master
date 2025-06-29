Welcome to the repository of my master thesis!

Here, you will find the code used to create samples and cDFT calculations for mixtures. Additionally, you will find the code used for building the feedforward neural networks through a MIMO and MISO approach, as well as the code used to create a continuous-time echo state network. Finally, the code for generating the figures in the thesis is also upladed.

The structure of the repository is as follows:
1. WriteToCSV.jl: randomly creates n samples of pressure, temperature, x1 and x2 based on given boundaries
2. Point_cDFT.jl: uses these samples to calculate density profiles of a one-dimensional pore through cDFT calculations

The other folders are relatively straightforward in the sense that their names represent the task the perform. The folder "Density Profiles" gives the density profiles gained from Point_cDFT.jl, and "Source" lists the density profiles divided into training and test set for the FFNNs. 
