#include <iostream>
#include <cuda_runtime.h>


// Cuda function
__global__ void VectorsAdd(int *a, int *b, int *c, int N) {
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < N){
        c[i] = a[i] + b[i];
    }
}

int main(){
    int N = 1000; //Size of each vector
    int memory_size = N * sizeof(int); // N * 4 bytes

    //Create h_a, h_b and h_c in CPUs memory
    int *h_a = new int[N];
    int *h_b = new int[N];
    int *h_c = new int[N];

    // Alocate values in h_a, h_b and h_c
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    //Create d_a, d_b and d_c in GPUs memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, memory_size);
    cudaMalloc(&d_b, memory_size);
    cudaMalloc(&d_c, memory_size);

    //Transfer d_a, d_b <- h_a, h_b   (CPU <- GPU)
    cudaMemcpy(d_a, h_a, memory_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, memory_size, cudaMemcpyHostToDevice);

    //GPU kernel specifications
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch CUDA kernel
    VectorsAdd <<<blocksPerGrid, threadsPerBlock>>> (d_a, d_b, d_c, N);

    //Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        std:: cerr<<"CUDA error: " <<cudaGetErrorString(err) << std:: endl;
        return -1;
    }

    //Transfer h_c <- d_c   (CPU <- GPU)
    cudaMemcpy(h_c, d_c, memory_size, cudaMemcpyDeviceToHost);

    //Print first 10 elements of the result vector
    for (int i = 0; i < 10; i++) {
        std::cout <<"h_c["<< i << "] = " << h_c[i] << std::endl;
    }

    //Release GPU (device) memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    //Release CPU (host) memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}