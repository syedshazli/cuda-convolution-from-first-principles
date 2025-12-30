# bash script to run all tests fir correctness and check if output is correct.

# Compile code for all tests
nvcc testFiftyvyFiftyImg.cu -o fifty
nvcc testTenbyTenImage.cu -o ten
nvcc testHundredByHundredImg.cu -o hund
nvcc testLargerFilter.cu -o filt
nvcc testThousandByThousandImg.cu -o thous