#include <cuda_runtime.h>
#include "common.cuh"

#define BLOCK_DIM 32
#define IDX2C2D(i,j,w) ((i)*(w)+(j))
#define IDX2C4D(i,j,k,s,c,h,w) (((i)*(c)*(h)*(w))+((j)*(h)*(w))+((k)*(w))+(s))

__global__ void Conv2dForwardKernel(float *X, float *K, float *Y, int N, int H, int W, int C_in, int C_out, int KH, int KW, int stride);
__global__ void Conv2dBackwardKernelDX(float *K, float *Dout, float *DX, int N, int H, int W, int C_in, int C_out, int KH, int KW, int stride);
__global__ void Conv2dBackwardKernelDK(float *X, float *Dout, float *DK, int N, int H, int W, int C_in, int C_out, int KH, int KW, int stride);
float* Conv2d(float *X, float *kernel, int N, int H, int W, int C_in, int C_out, int KH, int KW, int stride, int M, int K, int N_O);
float* Conv2dBackwardDX(float *kernel, float *Dout, int N, int H, int W, int C_in, int C_out, int KH, int KW, int stride, int M, int K, int N_O);
float* Conv2dBackwardDK(float *X, float *Dout, int N, int H, int W, int C_in, int C_out, int KH, int KW, int stride, int M, int K, int N_O);
