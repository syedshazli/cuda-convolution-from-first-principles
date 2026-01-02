# bash script to run all tests fir correctness and check if output is correct.

# Compile code for all tests
echo "COMPILING ALL TESTS"
nvcc testFiftyvyFiftyImg.cu -o fifty
nvcc testTenbyTenImage.cu -o ten
nvcc testHundredByHundredImg.cu -o hund
nvcc testLargerFilter.cu -o filt
nvcc testThousandByThousandImg.cu -o thous

echo "RUNNING TEN X TEN TEST"
./ten

# TODO: Add ligic if output is incorrect?

echo "RUNNING FIFTY X FIFTY TEST"
./fity

echo "RUNNING HUNDRED X HUNDRED TEST"
./hund

echo "RUNNING THOUSAND X THOUSAND TEST"
./thous

echo "RUNNING LARGER FILTER TEST"
./filt

#TODO: Show how many tests pass/fail?
echo "ALL TESTS COMPLETED"