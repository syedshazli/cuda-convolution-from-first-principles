#include <iostream>
#include <chrono>

// this is a sliding window X matmul problem
/**
 *  CUDA NOTES
 *  We want to minimize data transfer between device and host
 *  should batch small data transfers into a large data transfer
 *
 *  (Don't worry for non optimized) use cudaHostAlloc for cpu memory that's accessible to device
 *
 * */

/**
 * Alt unoptimized matmul
 * Assumes tileDim = N
*/
#define tileDim 2
using namespace std;
 __global__ void convolution(int *image, int *filter, int *output,
                               int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += image[][(1+i)/N] filter[i*N+col];
    }
    output[threadIdx.x] = sum;
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


        // FIXME: Fix launch parameters
        //convolution<<<2,dim3(2,2)>>> (dev_image,dev_filter,dev_output, 5);

        convolution<<<1, 5>>> (dev_image,dev_filter,dev_output, 5);

        cudaMemcpy(output, dev_output, sizeof(output), cudaMemcpyDeviceToHost);

        for(int row  = 0; row <2; row++ ){

           for(int col = 0; col<4; col++){//c++ XD

                cout<<output[row][col]<<','<<' ';

        }
        cout<<endl;
    }
        cudaFree(dev_image);
        cudaFree(dev_output);
        cudaFree(dev_filter);
}