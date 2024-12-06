#include <cuda_runtime.h>
#include "../common.cuh"

__global__ void matMulKernel(float* A, float* B, float* C, int M, int K, int N);
__global__ void matMulKernelATranspose(float *A, float *B, float *C, int M, int K, int N);
__global__ void matMulKernelBTranspose(float *A, float *B, float *C, int M, int K, int N);
float* matMul(float *A, float *B, int M, int K, int N, bool AT, bool BT);