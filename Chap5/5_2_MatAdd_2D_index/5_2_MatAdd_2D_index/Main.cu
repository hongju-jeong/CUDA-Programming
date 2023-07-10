#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROW_SIZE 32
#define COL_SIZE 32

__global__ void matAdd_2D_index(float* _dA, float* _dB, float* _dC)
{
	unsigned int col = threadIdx.x;
	unsigned int row = threadIdx.y;
	unsigned int index = row * blockDim.x + col;

	_dC[index] = _dA[index] + _dB[index];
}

int main(void) {

	// set the matrix on host
	float A[ROW_SIZE][COL_SIZE] = { 0 };
	float B[ROW_SIZE][COL_SIZE] = { 0 };
	float C[ROW_SIZE][COL_SIZE] = { 0 };
	float hC[ROW_SIZE][COL_SIZE] = { 0 };

	// set memory space on the device memory
	int matSize = ROW_SIZE * COL_SIZE;
	float* dA = NULL;
	float* dB = NULL;
	float* dC = NULL;
	cudaMalloc(&dA, sizeof(float) * matSize); cudaMemset(dA, 0, sizeof(float) * matSize);
	cudaMalloc(&dB, sizeof(float) * matSize); cudaMemset(dB, 0, sizeof(float) * matSize);
	cudaMalloc(&dC, sizeof(float) * matSize); cudaMemset(dC, 0, sizeof(float) * matSize);

	for (int iRow = 0; iRow < ROW_SIZE; iRow++)
		for (int iCol = 0; iCol < COL_SIZE; iCol++) {
			A[iRow][iCol] = rand() % 100;
			B[iRow][iCol] = rand() % 100;
			C[iRow][iCol] = A[iRow][iCol] + B[iRow][iCol];
		}

	cudaMemcpy(dA, A, sizeof(float) * matSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeof(float) * matSize, cudaMemcpyHostToDevice);

	dim3 blockDim(COL_SIZE, ROW_SIZE);
	matAdd_2D_index << <1, blockDim >> > (dA, dB, dC);

	cudaMemcpy(hC, dC, sizeof(float) * matSize, cudaMemcpyDeviceToHost);

	bool isCorrect = true;
	for (int iRow= 0 ; iRow < ROW_SIZE; iRow++)
		for (int iCol = 0; iCol < COL_SIZE; iCol++) {
			if (hC[iRow][iCol] != C[iRow][iCol]) {
				isCorrect = false;
				break;
			}
		}

	if (isCorrect) printf("GPU works well!\n");
	else printf("GPU fail to make correct resutl(s)..\n");


	return 0;
}