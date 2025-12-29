# bash script to run all tests fir correctnessvabd check if output is correct.
nvcc convolution.cu -o conv
g++ serialConvolution.cpp -o cpu