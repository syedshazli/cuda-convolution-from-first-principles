// This file shows what slidingWindow looks like with a filter that does not slide down any rows (Reaches end of some rows and moves down to the rest)
#include <iostream>
#include <chrono>

using namespace std;
 __global__ void slidingWindow(int *image, int *filter, int *output,
                               int imageWidth, int filterWidth, int filterHeight)
{
    int outputCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outputRow = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    for (int filterRow = 0; filterRow < filterHeight; filterRow++)
    {
        for(int filterCol = 0; filterCol < filterWidth; filterCol++)
        {
                int imageRow = outputRow + filterRow;
                int imageCol = outputCol + filterCol;

                sum += image[filterRow*imageWidth + outputCol + filterCol] * filter[filterRow * filterWidth + filterCol];
        }
    }

    output[outputRow * outputCol + outputCol] = sum;
}


int main(){
        int image[2][6] = {
        0, 2, 4, 6, 8, 10,
        3, 5, 7, 9, 11, 13
    };

         int filter[2][2] = {
        2, 1,
        1, 0
    };

        int output[1][5];

        int (*dev_output);// points to the first row of the array

        // allocate with CUDA MALLOC
        cudaMalloc( (void**) &dev_output, sizeof(output));

        int(*dev_image);

        int(*dev_filter);

        
        cudaMalloc( (void**) &dev_image, sizeof(image)  );
        cudaMalloc((void**) &dev_filter, sizeof(filter) );


        cudaMemcpy(dev_filter,filter,sizeof(filter),cudaMemcpyHostToDevice);
        cudaMemcpy(dev_image,image,sizeof(image),cudaMemcpyHostToDevice);


        int stride = 1;

        int filterLength = sizeof(filter)/sizeof(filter[0]);
        int filterWidth = sizeof(filter[0])/sizeof(filter[0][0]);

        int imageWidth = sizeof(filter[0])/sizeof(filter[0][0]);

        slidingWindow<<<1, 5>>> (dev_image,dev_filter,dev_output, imageWidth, filterLength, filterWidth);

        cudaMemcpy(output, dev_output, sizeof(output), cudaMemcpyDeviceToHost);

        for(int row  = 0; row <1; row++ ){

           for(int col = 0; col<5; col++){//c++ XD

                cout<<output[row][col]<<','<<' ';

        }
        cout<<endl;
    }
        cudaFree(dev_image);
        cudaFree(dev_output);
        cudaFree(dev_filter);
}