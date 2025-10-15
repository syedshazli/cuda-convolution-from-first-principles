#include <iostream>
#include <chrono>

using namespace std;
 __global__ void convolution(int *image, int *filter, int *output,
                               int N, int stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    int filterIdx = 0;

    for (int i = 0; i < N+2; i+= N+1) {

            sum += image[i+(tid*stride)] * filter[filterIdx];
            filterIdx += 1;
            sum += image[i+1+(tid*stride)] * filter[filterIdx];
            filterIdx +=1;

    }
    output[tid] = sum;
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

        int stride = 1;

        convolution<<<1, 5>>> (dev_image,dev_filter,dev_output, 5, stride);

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