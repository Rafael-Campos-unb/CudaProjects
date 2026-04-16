#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cuda/cmath>

//Define global function
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
	int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
	C[workIndex] = A[workIndex] + B[workIndex];
}

//Define auxiliary funcs
void initArray(float* arr, int n)
{
	for (int i = 0; i < n; i++)
	{
		arr[i] = (float)i;
	}
}

void SerialVecAdd(float* A, float* B, float* C, int n)
{
	for (int i = 0; i < n; i++)
	{
		C[i] = A[i] + B[i];
	}
}

bool vectorApproximatelyEqual(float* a, float* b, int n)
{
	for (int i = 0; i < n; i++)
	{
		float epsilon = 1e-5f;
		if (fabs(a[i] - b[i]) > epsilon)
		{
			return false;
		}
	}
	return true;
}

void ExplicityMemExample(int vectorLength)
{
	//Define vectors for Host 
	float* A = nullptr;
	float* B = nullptr;
	float* C = nullptr;
	float* comparisonResult = (float*)malloc(vectorLength * sizeof(float));

	//Define vector for device
	float* devA = nullptr;
	float* devB = nullptr;
	float* devC = nullptr;

	// Set memory space in CPU
	cudaMallocHost(&A, vectorLength * sizeof(float));
	cudaMallocHost(&B, vectorLength * sizeof(float));
	cudaMallocHost(&C, vectorLength * sizeof(float));




	// Initialize vectors in Host
	initArray(A, vectorLength);
	initArray(B, vectorLength);


	//Set memory space in GPU
	cudaMalloc(&devA, vectorLength * sizeof(float));
	cudaMalloc(&devB, vectorLength * sizeof(float));
	cudaMalloc(&devC, vectorLength * sizeof(float));

	//Copy data from CPU to GPU
	cudaMemcpy(devA, A, vectorLength * sizeof(float), cudaMemcpyDefault);
	cudaMemcpy(devB, B, vectorLength * sizeof(float), cudaMemcpyDefault);
	cudaMemset(devC, 0, vectorLength * sizeof(float));

	//Launch kernel
	int threads = 256;
	int blocks = cuda::ceil_div(vectorLength, threads);
	vecAdd << < blocks, threads >> > (devA, devB, devC, vectorLength);

	cudaDeviceSynchronize();

	//Return result of vector ad in GPU to CPU to compare
	cudaMemcpy(C, devC, vectorLength * sizeof(float), cudaMemcpyDefault);

	//Calculete serial vector ad in CPU to compare
	SerialVecAdd(A, B, comparisonResult, vectorLength);

	//Check if the results math
	if (vectorApproximatelyEqual(C, comparisonResult, vectorLength))
	{
		printf("Explicit Memory: respostas CPU e GPU convergiram\n");
	}
	else
	{
		printf("ExplicitMemory: respostas CPU e GPU não convergiram");
	}

	//clean up
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);
	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);
	free(comparisonResult);

}

int main()
{
	ExplicityMemExample(1024);
}
