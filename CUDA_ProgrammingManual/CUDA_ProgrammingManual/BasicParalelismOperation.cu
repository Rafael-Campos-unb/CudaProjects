
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cuda/cmath>


__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    C[workIndex] = A[workIndex] + B[workIndex];


}

int main()
{
    int threads = 256;
    int vectorLength = 1024;
    float* A, * B, * C;

    cudaMalloc(&A, vectorLength * sizeof(float));
    cudaMalloc(&B, vectorLength * sizeof(float));
    cudaMalloc(&C, vectorLength * sizeof(float));

    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd <<<blocks, threads>>> (A, B, C, vectorLength);
    cudaDeviceSynchronize();

    std::cout << "Calculo finalizado na GPU com sucesso!" << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}