#include <cuda_runtime.h>

__global__ void matMulKernel(float* A, float* B, float* C, int M, int K, int N);
float* matMul(float *A, float *B, int M, int K, int N);