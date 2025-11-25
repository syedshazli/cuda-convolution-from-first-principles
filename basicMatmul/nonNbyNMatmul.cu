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
#define tileDim 3
using namespace std;
 __global__ void matmul(int *a, int *b, int *c,
                               int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += a[row*tileDim+i] * b[i*N+col];
    }
    c[row*N+col] = sum;
}


int main(){
        int a[2][3] = {
        1, 2, 3,
        1, 2, 3
    };

         int b[3][4] = {
        5, 1, 2, 3,
        4, 3, 2, 1,
        3, 2, 1, 3,
    };

        // FIXME: infer the size of these data types, replace 2 and 4 with variables
        int c[2][4];

        int (*dev_c);// points to the first row of the array

        // allocate with CUDA MALLOC
        cudaMalloc( (void**) &dev_c, sizeof(c));

        int(*dev_a);

        int(*dev_b);

        cudaMalloc( (void**) &dev_a, sizeof(a)  );
        cudaMalloc((void**) &dev_b, sizeof(b) );


        cudaMemcpyAsync(dev_b,b,sizeof(b),cudaMemcpyHostToDevice);
        cudaMemcpyAsync(dev_a,a,sizeof(a),cudaMemcpyHostToDevice);

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        // 4 blocks, 2 threads per block
        matmul<<<dim3(2,1),dim3(2,2)>>> (dev_a,dev_b,dev_c, 4);

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << "[ns]" << std::endl;
        // finished computation, store result in dev_c
        cudaMemcpyAsync(c, dev_c, sizeof(c), cudaMemcpyDeviceToHost);

        for(int row  = 0; row <2; row++ ){

           for(int col = 0; col<4; col++){//c++ XD

                cout<<c[row][col]<<','<<' ';

        }
        cout<<endl;
    }
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);
}