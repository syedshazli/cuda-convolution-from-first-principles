#include <iostream>

// This is a known, working version on convolution, 
// and will be used to test performance speedups and correctness of our kernel
void convolution_cpu(int *image, int *filter, int *output,
                     int imageWidth, int imageHeight,
                     int filterWidth, int filterHeight) {
    int outputWidth = imageWidth - filterWidth + 1;
    int outputHeight = imageHeight - filterHeight + 1;
    
    for (int outRow = 0; outRow < outputHeight; outRow++) {
        for (int outCol = 0; outCol < outputWidth; outCol++) {
            int sum = 0;
            for (int fRow = 0; fRow < filterHeight; fRow++) {
                for (int fCol = 0; fCol < filterWidth; fCol++) {
                    int imgRow = outRow + fRow;
                    int imgCol = outCol + fCol;
                    sum += image[imgRow * imageWidth + imgCol] * 
                           filter[fRow * filterWidth + fCol];
                }
            }
            output[outRow * outputWidth + outCol] = sum;
        }
    }
}
int main()
{

    int image[3][6] = {
        0, 2, 4, 6, 8, 10,
        3, 5, 7, 9, 11,13,
        1, 2, 4, 7, 9, 12,
        };


         int filter[2][2] = {
        2, 1,
        1, 0
    };

    // TODO: Is this the right dimension?
    int output[1][5];
    int filterHeight = sizeof(filter)/sizeof(filter[0]);        
    int filterWidth = sizeof(filter[0])/sizeof(filter[0][0]);

    int imageWidth = sizeof(image[0])/sizeof(image[0][0]);   
    int imageHeight = sizeof(image)/sizeof(image[0]);     

    int outputLength = sizeof(output)/sizeof(output[0]);     
    int outputWidth = sizeof(output[0])/sizeof(output[0][0]);

   

    convolution_cpu((int*)image, (int*)filter, (int*)output, imageWidth, imageHeight, filterWidth, filterHeight);

    for(int row  = 0; row <outputLength; row++ ){

           for(int col = 0; col<outputWidth; col++){//c++ XD

                std::cout<<output[row][col]<<','<<' ';

        }
        std::cout<<std::endl;
    }
}