#include <cuda_runtime.h>
#include "../common.cuh"

__global__ void matAdd2dKernel(float *A, float *B, float *C, int N, int C_out);
__global__ void matAdd4dKernel(float *A, float *B, float *C, int N, int C_out, int OH, int OW);
float* matAdd2d(float *A, float *B, int N, int C_out);
float* matAdd4d(float *A, float *B, int N, int C_out, int OH, int OW);
