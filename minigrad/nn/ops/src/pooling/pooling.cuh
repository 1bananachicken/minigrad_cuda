#include <cuda_runtime.h>
#include "../common.cuh"

enum poolingType
{
    MAXPOOL,
    AVGPOOL,
};

__global__ void maxPool2dKernel(float *A, float *B, int *indices, int N, int C_in, int H, int W, int KH, int KW);
__global__ void avgPool2dKernel(float *A, float *B, int N, int C_in, int H, int W, int KH, int KW);
__global__ void avgPool2dBackwardKernel(float *Dout, float *DX, int N, int C_in, int H, int W, int KH, int KW);
std::tuple<float*, int*> pooling2d(float *A, int N, int C_in, int H, int W, int KH, int KW, poolingType ptype);
float* avgPool2dBackward(float *Dout, int N, int C_in, int H, int W, int KH, int KW);