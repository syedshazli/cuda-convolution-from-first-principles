#include <iostream>
#include <chrono>

// host must do cuda malloc on arrays a,b, c
// then a CUDAMEMCPY on array c
# define N 4
using namespace std;

__global__ void matmul(int a[4][4], int b[4][4], int c[4][4], int N){

	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	
	if(row<N && col < N)
	{
	int val = 0;
	for(int k = 0; k < N; k++){
		
		val += a[row][k] * b[k][col];	
	
	}
	
	c[row][col] = val;
	}

}

int main(){
	int a[4][4] = {
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
    };

	 int b[4][4] = {
        2, 2, 2, 2,
        2, 2, 2, 2,
        2, 2, 2, 2,
        2, 2, 2, 2,
    };

	int c[4][4];

	int (*dev_c)[4];// points to the first row of the array, each row has 4 ints
//	dev_c = c; // dev_c points to the 2d array c

	// allocate with CUDA MALLOC
	cudaMalloc( (void**) &dev_c, sizeof(c));
	
	int(*dev_a)[4];
	//dev_a = a;
	
	int(*dev_b)[4];
	//dev_b = b;

	cudaMalloc( (void**) &dev_a, sizeof(a)  );
	cudaMalloc((void**) &dev_b, sizeof(b) );
	
//	dev_a = a;
//	dev_b = b;

	cudaMemcpy(dev_b,b,sizeof(b),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_a,a,sizeof(a),cudaMemcpyHostToDevice);
	
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	// 4 blocks, 4 threads per block
	matmul<<<dim3(2,2),dim3(2,2)>>> (dev_a,dev_b,dev_c, 4);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << "[ns]" << std::endl;
	// finished computation, store result in dev_c
	cudaMemcpy(c, dev_c, sizeof(c), cudaMemcpyDeviceToHost);


	for(int row  = 0; row <4; row++ ){
		
     	   for(int col = 0; col<4; col++){//c++ XD

        	cout<<c[row][col]<<','<<' ';

        }
        cout<<endl;
    }
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);	
}