#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
using namespace std;


#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{

  if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

 __global__ void convolution(int *image, int *filter, int *output,
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

 
   sum += image[filterRow*imageWidth+ + outputCol + filterCol] * filter[filterRow * filterWidth + filterCol];
  }
    }

    output[outputRow * outputCol + outputCol] = sum;
}


int main(){
        int image[3][6] = {
        0, 2, 4, 6, 8, 10,
        3, 5, 7, 9, 11, 13,
        15, 17, 19, 21, 23
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

        CHECK_CUDA_ERROR(cudaMalloc( (void**) &dev_image, sizeof(image)));
        CHECK_CUDA_ERROR(cudaMalloc((void**) &dev_filter, sizeof(filter)));


        CHECK_CUDA_ERROR(cudaMemcpy(dev_filter,filter,sizeof(filter),cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(dev_image,image,sizeof(image),cudaMemcpyHostToDevice));


  int stride = 1;

  convolution<<<1, 5>>> (dev_image,dev_filter,dev_output, 6, 2, 2);

        CHECK_CUDA_ERROR(cudaMemcpy(output, dev_output, sizeof(output), cudaMemcpyDeviceToHost));

        int rowSize = sizeof(output)/sizeof(output[0]);
        int colSize = sizeof(output[0])/sizeof(output[0][0])

        for(int row  = 0; row <rowSize; row++ ){

           for(int col = 0; col<colSize; col++){

                cout<<output[row][col]<<','<<' ';

        }
        cout<<endl;
    }
        CHECK_CUDA_ERROR(cudaFree(dev_image));
        CHECK_CUDA_ERROR(cudaFree(dev_output));
        CHECK_CUDA_ERROR(cudaFree(dev_filter));
}