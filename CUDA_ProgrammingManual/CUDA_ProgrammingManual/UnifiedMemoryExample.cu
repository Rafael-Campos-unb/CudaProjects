#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cuda/cmath>

__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
	int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
	C[workIndex] = A[workIndex] + B[workIndex];

}


//Definir função initArray do host
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


void UnifiedMemExample(int vectorLength)
{
	//Pointers to memory
	float* A = nullptr;
	float* B = nullptr;
	float* C = nullptr;
	float* comparisonResult = (float*)malloc(vectorLength * sizeof(float));

	// Usando Unified Memory para alocar buffers
	cudaMallocManaged(&A, vectorLength);
	cudaMallocManaged(&B, vectorLength);
	cudaMallocManaged(&C, vectorLength);

	// Inicializar vetores no host
	initArray(A, vectorLength);
	initArray(B, vectorLength);

	// Lançar o kernel. Unified Memory garantirá que
	// os vetores A, B e C serão acessíveis para a GPU e a CPU
	int threads = 256;
	int blocks = cuda::ceil_div(vectorLength, threads);
	vecAdd <<<blocks, threads >>> (A, B, C, vectorLength);

	//Esperar o kernel completar a execução
	cudaDeviceSynchronize();

	//Rodar mesmo cálculo no host
	SerialVecAdd(A, B, comparisonResult, vectorLength);

	//Confirmando se os resultados da GPU e CPU convergem
	if (vectorApproximatelyEqual(C, comparisonResult, vectorLength))
	{
		printf("Unified Memory: respostas CPU e GPU convergiram\n ");
	}
	else
	{
		printf("Unified Memory: Erro - respostas CPU e GPU não convergiram");
	}

	//Limpando
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	free(comparisonResult);
	

}

int main()
{
	UnifiedMemExample(1024);

}